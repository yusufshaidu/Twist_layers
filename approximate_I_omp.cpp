#include <pybind11/pybind11.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>  // Include OpenMP header
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector<std::vector<int>> compute_approximate_I(
    double a_exact, double b_exact, double c_exact, double d_exact,
    int nmax, double tol) {

    std::vector<std::vector<int>> I;

    // Parallelize the outermost loops (i, j, k, l)
    #pragma omp parallel for collapse(4) shared(I)
    //#pragma omp parallel for collapse(1) shared(I)
    for (int i = -nmax; i <= nmax; ++i) {
        for (int j = -nmax; j <= nmax; ++j) {
            for (int k = -nmax; k <= nmax; ++k) {
                for (int l = -nmax; l <= nmax; ++l) {
                    double det = i * l - j * k;
                    if (std::abs(det) < 1e-6 || i == 0 || j == 0 || k == 0 || l == 0)
                        continue;
                    for (int m = -nmax; m <= nmax; ++m) {
                        for (int n = -nmax; n <= nmax; ++n) {
                            for (int q = -nmax; q <= nmax; ++q) {
                                for (int r = -nmax; r <= nmax; ++r) {
                                    double det2 = m * r - n * q;
                                    if (std::abs(det2) < 1e-6 || m == 0 || n == 0 || q == 0 || r == 0)
                                        continue;

                                    double a = (l * m - j * q) / det;
                                    double b = (l * n - j * r) / det;
                                    double c = (-k * m + i * q) / det;
                                    double d = (-k * n + i * r) / det;

                                    double error = std::max({std::abs(a - a_exact), std::abs(b - b_exact),
                                                             std::abs(c - c_exact), std::abs(d - d_exact)});

                                    if (error < tol) {
                                        #pragma omp critical
                                        I.push_back({i, j, k, l, m, n, q, r});
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return I;
}

// Pybind11 bindings
PYBIND11_MODULE(approximate_I_omp_module, m) {
    m.def("compute_approximate_I", &compute_approximate_I, 
          "Compute the approximate ijkl, mnqr satisfying the commensurability condition with OpenMP parallelism",
          py::arg("a_exact"), py::arg("b_exact"), py::arg("c_exact"), py::arg("d_exact"),
          py::arg("nmax"), py::arg("tol"));
}
