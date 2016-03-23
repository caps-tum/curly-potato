// compile with C++ 14 (for auto return type deduction) + OpenMP
// e.g. clang++ -g3 -std=c++14 -fopenmp space.cpp

#include <functional>
#include <iostream>
#include <tuple>
#include <type_traits>

#include <cassert>

#include <omp.h>

// prototype for N-DIM iteration
template <int DIM, typename space> struct iteration;

// simple 2-D iteration
template <typename spaceT> struct iteration<2, spaceT> {
	int i, j;

	std::function<void(int &, int &, spaceT)> _order;

	spaceT _space;

	iteration<2, spaceT>() = delete;
	iteration<2, spaceT>(const iteration<2, spaceT> &) = default;

	iteration<2, spaceT>(spaceT s) : i(s.start[0]), j(s.start[1]), _space(s) {}

	bool operator!=(const iteration<2, spaceT> &rhs) const noexcept {
		return rhs.i != i || rhs.j != j || rhs._space != _space;
	}

	void operator++() noexcept {
		assert(_order);
		_order(i, j, _space);
	}

	auto operator*() const noexcept { return std::make_tuple(i, j); }
};

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
	static void next(int &i, int &j, spaceT space) {
		++i;
		if (i >= space.end[0]) {
			i = space.start[0];
			++j;
			if (j >= space.end[1]) {
				i = space.end[0];
				j = space.end[1];
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
		temp.i = _space.end[0];
		temp.j = _space.end[1];
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
