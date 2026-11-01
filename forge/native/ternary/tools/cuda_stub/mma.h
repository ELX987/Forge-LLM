#pragma once
#include "cuda_runtime.h"
namespace nvcuda { namespace wmma {
struct matrix_a {}; struct matrix_b {}; struct accumulator {}; struct row_major {}; struct col_major {};
enum layout_t { mem_row_major, mem_col_major };
template <class Use, int M, int N, int K, class T, class L = void> struct fragment { T dummy[8]; };
template <class F, class V> void fill_fragment(F&, V);
template <class Use, int M, int N, int K, class T, class L> void load_matrix_sync(fragment<Use, M, N, K, T, L>&, const T*, unsigned);
template <class F, class T> void store_matrix_sync(T*, const F&, unsigned, layout_t);
template <class D, class A, class B, class C> void mma_sync(D&, const A&, const B&, const C&);
} }
