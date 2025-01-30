#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#ifdef USE_BL
#include "plugin-PixelTriplets/BrokenLine.h"
#else
#include "plugin-PixelTriplets/RiemannFit.h"
#endif

#include "test_common.h"

using namespace Eigen;

namespace Rfit {
  constexpr uint32_t maxNumberOfTracks() { return 5 * 1024; }
  constexpr uint32_t stride() { return maxNumberOfTracks(); }
  // hits
  template <int N>
  using Matrix3xNd = Eigen::Matrix<double, 3, N>;
  template <int N>
  using Map3xNd = Eigen::Map<Matrix3xNd<N>, 0, Eigen::Stride<3 * stride(), stride()> >;
  // errors
  template <int N>
  using Matrix6xNf = Eigen::Matrix<float, 6, N>;
  template <int N>
  using Map6xNf = Eigen::Map<Matrix6xNf<N>, 0, Eigen::Stride<6 * stride(), stride()> >;
  // fast fit
  using Map4d = Eigen::Map<Vector4d, 0, Eigen::InnerStride<stride()> >;

}  // namespace Rfit

/*
Hit global: 641,0 2: 2.934787,0.773211,-10.980247
Error: 641,0 2: 1.424715e-07,-4.996975e-07,1.752614e-06,3.660689e-11,1.644638e-09,7.346080e-05
Hit global: 641,1 104: 6.314229,1.816356,-23.162731
Error: 641,1 104: 6.899177e-08,-1.873414e-07,5.087101e-07,-2.078806e-10,-2.210498e-11,4.346079e-06
Hit global: 641,2 1521: 8.936963,2.765734,-32.759060
Error: 641,2 1521: 1.406273e-06,4.042467e-07,6.391180e-07,-3.141497e-07,6.513821e-08,1.163863e-07
Hit global: 641,3 1712: 10.360559,3.330824,-38.061260
Error: 641,3 1712: 1.176358e-06,2.154100e-07,5.072816e-07,-8.161219e-08,1.437878e-07,5.951832e-08
Hit global: 641,4 1824: 12.856387,4.422212,-47.518867
Error: 641,4 1824: 2.852843e-05,7.956492e-06,3.117701e-06,-1.060541e-06,8.777413e-09,1.426417e-07
*/

template <typename M3xN, typename M6xN>
/**
 * @brief Fills hits and hits covariance matrices with predefined values.
 *
 * This function populates the hits matrix and hits covariance matrix with
 * hardcoded values depending on the number of columns in the input matrix.
 *
 * @param[out] hits The hits matrix to be filled.
 * @param[out] hits_ge The hits covariance matrix to be filled.
 */
// The above comment was written by an LLM. 
void fillHitsAndHitsCov(M3xN& hits, M6xN& hits_ge) {
  constexpr uint32_t N = M3xN::ColsAtCompileTime;

  if (N == 5) {
    hits << 2.934787, 6.314229, 8.936963, 10.360559, 12.856387, 0.773211, 1.816356, 2.765734, 3.330824, 4.422212,
        -10.980247, -23.162731, -32.759060, -38.061260, -47.518867;
    hits_ge.col(0) << 1.424715e-07, -4.996975e-07, 1.752614e-06, 3.660689e-11, 1.644638e-09, 7.346080e-05;
    hits_ge.col(1) << 6.899177e-08, -1.873414e-07, 5.087101e-07, -2.078806e-10, -2.210498e-11, 4.346079e-06;
    hits_ge.col(2) << 1.406273e-06, 4.042467e-07, 6.391180e-07, -3.141497e-07, 6.513821e-08, 1.163863e-07;
    hits_ge.col(3) << 1.176358e-06, 2.154100e-07, 5.072816e-07, -8.161219e-08, 1.437878e-07, 5.951832e-08;
    hits_ge.col(4) << 2.852843e-05, 7.956492e-06, 3.117701e-06, -1.060541e-06, 8.777413e-09, 1.426417e-07;
    return;
  }

  if (N > 3)
    hits << 1.98645, 4.72598, 7.65632, 11.3151, 2.18002, 4.88864, 7.75845, 11.3134, 2.46338, 6.99838, 11.808, 17.793;
  else
    hits << 1.98645, 4.72598, 7.65632, 2.18002, 4.88864, 7.75845, 2.46338, 6.99838, 11.808;

  hits_ge.col(0)[0] = 7.14652e-06;
  hits_ge.col(1)[0] = 2.15789e-06;
  hits_ge.col(2)[0] = 1.63328e-06;
  if (N > 3)
    hits_ge.col(3)[0] = 6.27919e-06;
  hits_ge.col(0)[2] = 6.10348e-06;
  hits_ge.col(1)[2] = 2.08211e-06;
  hits_ge.col(2)[2] = 1.61672e-06;
  if (N > 3)
    hits_ge.col(3)[2] = 6.28081e-06;
  hits_ge.col(0)[5] = 5.184e-05;
  hits_ge.col(1)[5] = 1.444e-05;
  hits_ge.col(2)[5] = 6.25e-06;
  if (N > 3)
    hits_ge.col(3)[5] = 3.136e-05;
  hits_ge.col(0)[1] = -5.60077e-06;
  hits_ge.col(1)[1] = -1.11936e-06;
  hits_ge.col(2)[1] = -6.24945e-07;
  if (N > 3)
    hits_ge.col(3)[1] = -5.28e-06;
}

template <int N>
/**
 * @brief Tests the fitting functions with generated hits
 
void testFit()
 * @brief Fills the hits matrix and covariance matrix
 * @param[out] hits The hits matrix
 * @param[out] hits_ge The covariance matrix of hits
 
void fillHitsAndHitsCov(Rfit::Matrix3xNd<N> &hits, Rfit::Matrix6xNf<N> &hits_ge)
 * @brief Performs a fast fit on the given hits
 * @param[in] hits The hits matrix
 * @param[out] results The fitted parameters
 
void Fast_fit(const Rfit::Matrix3xNd<N> &hits, Vector4d &results)
 * @brief Prepares broken line data from hits and initial fit results
 * @param[in] hits The hits matrix
 * @param[in] initResults The initial fit results
 * @param[in] B The curvature parameter
 * @param[out] data The prepared broken line data
 
void prepareBrokenLineData(const Rfit::Matrix3xNd<N> &hits, const Vector4d &initResults, double B, PreparedBrokenLineData<N> &data)
 * @brief Performs a line fit on the given hits and initial fit results
 * @param[in] hitsGe The covariance matrix of hits
 * @param[in] initResults The initial fit results
 * @param[in] B The curvature parameter
 * @param[in] data The prepared broken line data
 * @param[out] results The fitted line parameters
 
void BL_Line_fit(const Rfit::Matrix6xNf<N> &hitsGe, const Vector4d &initResults, double B, const PreparedBrokenLineData<N> &data, line_fit &results)
 * @brief Performs a circle fit on the given hits and initial fit results
 * @param[in] hits The hits matrix
 * @param[in] hitsGe The covariance matrix of hits
 * @param[in] initResults The initial fit results
 * @param[in] B The curvature parameter
 * @param[in] data The prepared broken line data
 * @param[out] results The fitted circle parameters
 
void BL_Circle_fit(const Rfit::Matrix3xNd<N> &hits, const Rfit::Matrix6xNf<N> &hitsGe, const Vector4d &initResults, double B, const PreparedBrokenLineData<N> &data, karimaki_circle_fit &results)
 * @brief Loads the 2D covariance matrix from the given hits and covariance matrix
 * @param[in] hitsGe The covariance matrix of hits
 * @param[out] hitsCov The loaded 2D covariance matrix
 
void loadCovariance2D(const Rfit::Matrix6xNf<N> &hitsGe, Rfit::Matrix2Nd<N> &hitsCov)
 * @brief Performs a circle fit on the given hits and initial fit results
 * @param[in] hits The hits matrix
 * @param[in] hitsCov The covariance matrix of hits
 * @param[in] initResults The initial fit results
 * @param[in] rad The radii of the hits
 * @param[in] B The curvature parameter
 * @param[in] useUncertainties Whether to use uncertainties in the fit
 * @return The fitted circle parameters
 
Rfit::circle_fit Circle_fit(const Rfit::Matrix2Nd<N> &hits, const Rfit::Matrix2Nd<N> &hitsCov, const Vector4d &initResults, const Rfit::VectorNd<N> &rad, double B, bool useUncertainties)
 * @brief Performs a line fit on the given hits and initial fit results
 * @param[in] hits The hits matrix
 * @param[in] hitsGe The covariance matrix of hits
 * @param[in] circleResults The fitted circle parameters
 * @param[in] initResults The initial fit results
 * @param[in] B The curvature parameter
 * @param[in] useUncertainties Whether to use uncertainties in the fit
 * @return The fitted line parameters
 
Rfit::line_fit Line_fit(const Rfit::Matrix3xNd<N> &hits, const Rfit::Matrix6xNf<N> &hitsGe, const Rfit::circle_fit &circleResults, const Vector4d &initResults, double B, bool useUncertainties)
 * @brief Converts the fitted circle parameters to par-uv-r-tau format
 * @param[inout] results The fitted circle parameters
 * @param[in] B The curvature parameter
 * @param[in] useUncertainties Whether to use uncertainties in the conversion
 
void par_uvrtopak(Rfit::circle_fit &results, double B, bool useUncertainties)*/
// The above comment was written by an LLM. 
void testFit() {
  constexpr double B = 0.0113921;
  Rfit::Matrix3xNd<N> hits;
  Rfit::Matrix6xNf<N> hits_ge = MatrixXf::Zero(6, N);

  fillHitsAndHitsCov(hits, hits_ge);

  std::cout << "sizes " << N << ' ' << sizeof(hits) << ' ' << sizeof(hits_ge) << ' ' << sizeof(Vector4d) << std::endl;

  std::cout << "Generated hits:\n" << hits << std::endl;
  std::cout << "Generated cov:\n" << hits_ge << std::endl;

  // FAST_FIT_CPU
  Vector4d fast_fit_results;
#ifdef USE_BL
  BrokenLine::BL_Fast_fit(hits, fast_fit_results);
#else
#ifndef DISABLE_RFIT
  Rfit::Fast_fit(hits, fast_fit_results);
#endif
#endif
  std::cout << "Fitted values (FastFit, [X0, Y0, R, tan(theta)]):\n" << fast_fit_results << std::endl;

  // CIRCLE_FIT CPU

#ifdef USE_BL
  BrokenLine::PreparedBrokenLineData<N> data;
  BrokenLine::karimaki_circle_fit circle_fit_results;
  Rfit::Matrix3d Jacob;

  BrokenLine::prepareBrokenLineData(hits, fast_fit_results, B, data);
  Rfit::line_fit line_fit_results;
  BrokenLine::BL_Line_fit(hits_ge, fast_fit_results, B, data, line_fit_results);
  BrokenLine::BL_Circle_fit(hits, hits_ge, fast_fit_results, B, data, circle_fit_results);
  Jacob << 1., 0, 0, 0, 1., 0, 0, 0,
      -B / std::copysign(Rfit::sqr(circle_fit_results.par(2)), circle_fit_results.par(2));
  circle_fit_results.par(2) = B / std::abs(circle_fit_results.par(2));
  circle_fit_results.cov = Jacob * circle_fit_results.cov * Jacob.transpose();
#else
  Rfit::circle_fit circle_fit_results;
  Rfit::line_fit line_fit_results;
#ifndef DISABLE_RFIT
  Rfit::VectorNd<N> rad = (hits.block(0, 0, 2, N).colwise().norm());
  Rfit::Matrix2Nd<N> hits_cov = Rfit::Matrix2Nd<N>::Zero();
  Rfit::loadCovariance2D(hits_ge, hits_cov);
  circle_fit_results = Rfit::Circle_fit(hits.block(0, 0, 2, N), hits_cov, fast_fit_results, rad, B, true);
  // LINE_FIT CPU
  line_fit_results = Rfit::Line_fit(hits, hits_ge, circle_fit_results, fast_fit_results, B, true);
  Rfit::par_uvrtopak(circle_fit_results, B, true);
#endif
#endif

  std::cout << "Fitted values (CircleFit):\n"
            << circle_fit_results.par << "\nchi2 " << circle_fit_results.chi2 << std::endl;
  std::cout << "Fitted values (LineFit):\n" << line_fit_results.par << "\nchi2 " << line_fit_results.chi2 << std::endl;

  std::cout << "Fitted cov (CircleFit) CPU:\n" << circle_fit_results.cov << std::endl;
  std::cout << "Fitted cov (LineFit): CPU\n" << line_fit_results.cov << std::endl;
}

/**
 * Main program entry point 
 */
// The above comment was written by an LLM. 
int main(int argc, char* argv[]) {
  testFit<4>();
  testFit<3>();
  testFit<5>();

  return 0;
}
