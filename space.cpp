// compile with C++ 14 (for auto return type deduction) + OpenMP
// e.g. clang++ -g3 -std=c++14 -fopenmp space.cpp

#include <functional>
#include <iostream>
#include <tuple>
#include <type_traits>

#include <cassert>

#include <omp.h>

template <int N, typename T> struct array_to_tuple {
	static inline auto get(const T &arr) noexcept {
		return std::tuple_cat(std::make_tuple(arr[std::tuple_size<T>::value - N]), array_to_tuple<N - 1, T>::get(arr));
	}
};

template <typename T> struct array_to_tuple<1, T> {
	static inline auto get(const T &arr) noexcept { return std::make_tuple(arr[std::tuple_size<T>::value - 1]); }
};

template <int DIM, typename spaceT> struct iteration {
	std::array<int, DIM> index;

	std::function<void(std::array<int, DIM> &, spaceT)> _order;

	spaceT _space;

	iteration<DIM, spaceT>() = delete;
	iteration<DIM, spaceT>(const iteration<DIM, spaceT> &) = default;

	iteration<DIM, spaceT>(spaceT s) : index(s.start), _space(s) {}

	bool operator!=(const iteration<DIM, spaceT> &rhs) const noexcept {
		return rhs.index != index || rhs._space != _space;
	}

	void operator++() noexcept {
		assert(_order);
		_order(index, _space);
	}

	auto operator*() const noexcept { return array_to_tuple<DIM, decltype(index)>::get(index); }
};

// prevent anyone from using a space with 0 dimension
template <typename spaceT> struct iteration<0, spaceT>;

// _dense_space<1> * _dense_space<1> = _dense_space<2>?
template <int DIM> struct _dense_space {
	std::array<int, DIM> start, end;
	static constexpr int dim = DIM;

	template <typename... argsT> _dense_space(const int _start, const int _end, argsT... args) {
		static_assert(sizeof...(args) == (DIM - 1) * 2, "Missing constructor parameters for dense_space.");
		start[0] = _start;
		end[0] = _end;
		init<DIM - 1>(std::forward<argsT>(args)...);
	}

	bool operator!=(const _dense_space<DIM> &rhs) const noexcept { return rhs.start != start || rhs.end != end; }

  private:
	template <int N, typename... argsT> inline void init(const int _start, const int _end, argsT... args) {
		static_assert(sizeof...(args) == (N - 1) * 2, "Internal error. Something is broken with our constructor.");
		start[DIM - N] = _start;
		end[DIM - N] = _end;
	}
};

// just a little helper
template <typename... argsT> auto dense_space(argsT... args) {
	return _dense_space<sizeof...(argsT) / 2>(std::forward<argsT>(args)...);
}

template <typename spaceT> struct _rm_order {
	spaceT _space;

	_rm_order(spaceT s) : _space(s) {}

	// TODO generalize it, only ok if spaceT === space2d
	static void next(std::array<int, 2> &arr, spaceT space) {
		++arr[0];
		if (arr[0] >= space.end[0]) {
			arr[0] = space.start[0];
			++arr[1];
			if (arr[1] >= space.end[1]) {
				arr[0] = space.end[0];
				arr[1] = space.end[1];
			}
		}
	}

	auto begin() const noexcept {
		// FIXME infere dimension from spaceT
		auto temp = iteration<2, spaceT>(_space);
		temp._order = next;

		return temp;
	}

	auto end() const noexcept {
		// FIXME infere dimension from spaceT
		auto temp = iteration<2, spaceT>(_space);
		temp.index[0] = _space.end[0];
		temp.index[1] = _space.end[1];
		temp._order = next;

		return temp;
	}
};

// just a little helper
template <typename T> auto rm_order(T &&instance) { return _rm_order<T>(std::forward<T>(instance)); }

// TODO should allow to select the dimonsion used to partition
template <typename spaceT> struct _static_partition : public spaceT {
	_static_partition() = delete;

	_static_partition(const int dim, spaceT o) : spaceT(o) {
		int id = omp_get_thread_num();
		int threads = omp_get_num_threads();
		int size = spaceT::end[dim] - spaceT::start[dim];
		spaceT::start[dim] = (size / threads) * id;
		if (id != threads - 1) spaceT::end[dim] = size / threads * (id + 1);
	}
};

// just a little helper
template <typename T> auto static_partition(const int dim, T &&instance) {
	return _static_partition<T>(dim, std::forward<T>(instance));
}

int main(int argc, char const *argv[]) {

	double arr1[100][100], arr2[100][100];

#pragma omp parallel
	{
		int i, j;
		for (const auto &iteration : rm_order(static_partition(0, dense_space(1, 99, 1, 99)))) {
			std::tie(i, j) = std::move(iteration);
			arr1[i][j] = (arr2[i - 1][j] + arr2[i + 1][j] + arr2[i][j - 1] + arr2[i][j + 1]) / 4;
		}
	}
	return 0;
}
