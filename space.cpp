// compile with C++ 14 (for auto return type deduction) + OpenMP
// e.g. clang++ -g3 -std=c++14 -fopenmp space.cpp

#include <functional>
#include <iostream>
#include <tuple>
#include <type_traits>

#include <cassert>

#include <omp.h>

namespace impl {
template <int N, typename T> struct array_to_tuple {
	static inline auto get(const T &arr) noexcept {
		return std::tuple_cat(std::make_tuple(arr[std::tuple_size<T>::value - N]), array_to_tuple<N - 1, T>::get(arr));
	}
};

template <typename T> struct array_to_tuple<1, T> {
	static inline auto get(const T &arr) noexcept { return std::make_tuple(arr[std::tuple_size<T>::value - 1]); }
};
}

template <int DIM, typename spaceT> struct iteration {
	std::array<int, DIM> index;
	std::function<void(decltype(index) &, spaceT)> order;

	spaceT _space;

	iteration<DIM, spaceT>() = delete;
	iteration<DIM, spaceT>(const iteration<DIM, spaceT> &) = default;

	iteration<DIM, spaceT>(const spaceT &s) : index(s.start), _space(s) {}

	bool operator!=(const iteration<DIM, spaceT> &rhs) const noexcept {
		return rhs.index != index || rhs._space != _space;
	}

	void operator++() noexcept {
		assert(order);
		order(index, _space);
	}

	auto operator*() const noexcept { return impl::array_to_tuple<DIM, decltype(index)>::get(index); }
};

// prevent anyone from using a space with 0 dimension
template <typename spaceT> struct iteration<0, spaceT>;

// _dense_space<1> * _dense_space<1> = _dense_space<2>?
template <int DIM> struct _dense_space {
	std::array<int, DIM> start, limit;
	static constexpr int dim = DIM;

	template <typename... argsT> _dense_space(const int _start, const int _limit, argsT... args) {
		static_assert(sizeof...(args) == (DIM - 1) * 2, "Missing constructor parameters for dense_space.");
		start[0] = _start;
		limit[0] = _limit;
		init<DIM - 1>(std::forward<argsT>(args)...);
	}

	bool operator!=(const _dense_space<DIM> &rhs) const noexcept { return rhs.start != start || rhs.limit != limit; }

	auto begin() const noexcept {
		auto temp = iteration<DIM, _dense_space<DIM>>(*this);
		temp.index = start;
		return temp;
	}
	auto end() const noexcept {
		auto temp = iteration<DIM, _dense_space<DIM>>(*this);
		temp.index = limit;
		return temp;
	}

  private:
	template <int N, typename... argsT> inline void init(const int _start, const int _end, argsT... args) {
		static_assert(sizeof...(args) == (N - 1) * 2, "Internal error. Something is broken with our constructor.");
		start[DIM - N] = _start;
		limit[DIM - N] = _end;
	}
};

// just a little helper
template <typename... argsT> auto dense_space(argsT... args) {
	return _dense_space<sizeof...(argsT) / 2>(std::forward<argsT>(args)...);
}

namespace impl {
template <int N, typename T, typename spaceT> struct cm_next {
	static inline void get(decltype(spaceT::start) &arr, const spaceT &space) noexcept {
		constexpr int index = spaceT::dim - N;
		++arr[index];
		if (arr[index] >= space.limit[index]) {
			arr[index] = space.start[index];
			impl::cm_next<N - 1, T, spaceT>::get(arr, space);
		}
	}
};

template <typename T, typename spaceT> struct cm_next<1, T, spaceT> {
	static inline void get(decltype(spaceT::start) &arr, const spaceT &space) noexcept {
		constexpr int index = spaceT::dim - 1;
		++arr[index];
		if (arr[index] >= space.limit[index]) {
			arr = space.limit;
		}
	}
};
}

template <typename spaceT> struct _cm_order {
	spaceT _space; // could be a partitioned space

	_cm_order(spaceT s) : _space(s) {}

	static void next(decltype(spaceT::start) &arr, const spaceT &space) noexcept {
		impl::cm_next<spaceT::dim, decltype(spaceT::start), spaceT>::get(arr, space);
	}

	auto begin() const noexcept {
		iteration<spaceT::dim, spaceT> temp(_space);
		temp.order = next;

		return temp;
	}

	auto end() const noexcept {
		iteration<spaceT::dim, spaceT> temp(_space);
		temp.index = _space.limit;
		temp.order = next;

		return temp;
	}
};

// just a little helper
template <typename T> auto cm_order(T &&instance) { return _cm_order<T>(std::forward<T>(instance)); }

template <typename spaceT> struct _static_partition : public spaceT {
	_static_partition() = delete;

	_static_partition(const int dim, spaceT o) : spaceT(o) {
		int id = omp_get_thread_num();
		int threads = omp_get_num_threads();
		int size = spaceT::limit[dim] - spaceT::start[dim];
		spaceT::start[dim] = (size / threads) * id;
		if (id != threads - 1) spaceT::limit[dim] = size / threads * (id + 1);
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
		for (const auto &iteration : cm_order(static_partition(0, dense_space(1, 9, 1, 9)))) {
			std::tie(i, j) = std::move(iteration);
#pragma omp critical
			std::cout << omp_get_thread_num() << " - " << i << " - " << j << std::endl;
			arr1[i][j] = (arr2[i - 1][j] + arr2[i + 1][j] + arr2[i][j - 1] + arr2[i][j + 1]) / 4;
		}
	}
	return 0;
}
