#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.6;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  n_aug_ = 7;

  lambda_ = 3 - n_aug_;

  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(0.5/(n_aug_ + lambda_));
  weights_(0) = lambda_/(lambda_ + n_aug_); // special first weight

  time_us_ = 0;

  is_initialized_ = false;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  if((meas_package.sensor_type_ == MeasurementPackage::LASER && this->use_laser_ == false) ||
     (meas_package.sensor_type_ == MeasurementPackage::RADAR && this->use_radar_ == false)) {
    return;
  }

  if(this->is_initialized_ == false) {
    this->x_ << 1, 1, 1, 0.0, 0.0;
    
    if(meas_package.sensor_type_ == MeasurementPackage::LASER) {
      this->x_ = VectorXd(this->n_x_);
      this->x_(0) = meas_package.raw_measurements_(0);
      this->x_(1) = meas_package.raw_measurements_(1);
    }
    else { //MeasurementPackage::RADAR
      this->x_ = VectorXd(this->n_x_);
      this->x_(0) = meas_package.raw_measurements_(0) * cos(meas_package.raw_measurements_(1));
      this->x_(1) = meas_package.raw_measurements_(0) * sin(meas_package.raw_measurements_(1));
    }
    this->is_initialized_ = true;

    this->P_ = MatrixXd(this->n_x_, this->n_x_);
    this->P_.fill(0);
    this->P_(0,0) = 0.2;
    this->P_(1,1) = 0.2;
    this->P_(2,2) = 1.0;
    this->P_(3,3) = 1.0;
    this->P_(4,4) = 1.0;
  }
  else {
    double dt = (meas_package.timestamp_ - this->time_us_) / 1000000.0;
    if(dt > 0.0001) {
      this->Prediction(dt);

      if(meas_package.sensor_type_ == MeasurementPackage::LASER) {
        this->UpdateLidar(meas_package);
      }
      else if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        this->UpdateRadar(meas_package);
      }
    }
  }

  this->time_us_ = meas_package.timestamp_;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  MatrixXd Xsig_aug;
  this->GenerateAugmentedSigmaPoints(&Xsig_aug);
  this->UpdateSigmaPointPrediction(delta_t, Xsig_aug);
  this->PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  MatrixXd Zsig_pred;
  VectorXd Zpred;
  MatrixXd Spred;
  this->PredictLidarMeasurement(&Zsig_pred, &Zpred, &Spred);

  this->UpdateState(Zsig_pred, Zpred, Spred, meas_package);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  MatrixXd Zsig_pred;
  VectorXd Zpred;
  MatrixXd Spred;
  this->PredictRadarMeasurement(&Zsig_pred, &Zpred, &Spred);

  this->UpdateState(Zsig_pred, Zpred, Spred, meas_package);
}

double UKF::NormalizeAngle(double angle) {
  while (angle > M_PI) {
    if(angle > 2 * M_PI) {
      angle = fmod(angle, ( 2 * M_PI));
    }
    else {
      angle -=2.*M_PI;
    }
  }
  while (angle < -1.0 * M_PI){
    if(angle < -2 * M_PI) {
      angle = fmod(angle, ( -2 * M_PI));
    }
    else {
      angle += 2.0 * M_PI;
    }
  }
  return angle;
}

void UKF::GenerateAugmentedSigmaPoints(MatrixXd* Xsig_aug_out) {

  //create augmented mean vector
  VectorXd x_aug = VectorXd(this->n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(this->n_aug_, this->n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(this->n_aug_, 2 * this->n_aug_ + 1);

  //create augmented mean state
  x_aug.fill(0.0);
  x_aug.head(this->n_x_) = this->x_;

  //create augmented covariance matrix
  P_aug.fill(0);
  P_aug.topLeftCorner(this->n_x_,this->n_x_) = this->P_;
  P_aug(this->n_x_,this->n_x_) = this->std_a_ * this->std_a_;
  P_aug(this->n_x_ + 1,this->n_x_ + 1) = this->std_yawdd_ * this->std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sig
  //yaw = NormalizeAngle(yaw);ma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< this->n_aug_; i++) {
    VectorXd offset = sqrt(this->lambda_ + this->n_aug_) * L.col(i);
    Xsig_aug.col(i+1)                = x_aug + offset;
    Xsig_aug.col(i+1 + this->n_aug_) = x_aug - offset;
  }

  //write result
  *Xsig_aug_out = Xsig_aug;
}

void UKF::UpdateSigmaPointPrediction(double delta_t, MatrixXd& Xsig_aug) {

  //create matrix with predicted sigma points as columns
  this->Xsig_pred_ = MatrixXd(this->n_x_, 2 * this->n_aug_ + 1);
  this->Xsig_pred_.fill(0.0);

  //predict sigma points
  for (int i = 0; i< 2 * this->n_aug_ + 1; i++) {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);

    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin(yaw + yawd*delta_t) - sin(yaw) );
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw + yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    this->Xsig_pred_(0,i) = px_p;
    this->Xsig_pred_(1,i) = py_p;
    this->Xsig_pred_(2,i) = v_p;
    this->Xsig_pred_(3,i) = yaw_p;
    this->Xsig_pred_(4,i) = yawd_p;
  }

}

void UKF::PredictMeanAndCovariance() {

  //predicted state mean
  this->x_.fill(0.0);
  for (int i = 0; i < 2 * this->n_aug_ + 1; i++) {
    this->x_ = this->x_ + this->weights_(i) * this->Xsig_pred_.col(i);
  }
  this->x_(3) = NormalizeAngle(this->x_(3));
  this->x_(4) = NormalizeAngle(this->x_(4));

  //predicted state covariance matrix
  this->P_.fill(0.0);
  for (int i = 0; i < 2 * this->n_aug_ + 1; i++) {
    // state difference
    VectorXd x_diff = this->Xsig_pred_.col(i) - this->x_;

    //angle normalization
    x_diff(3) = NormalizeAngle(x_diff(3));
    x_diff(4) = NormalizeAngle(x_diff(4));

    this->P_ = this->P_ + this->weights_(i) * x_diff * x_diff.transpose() ;
  }

}

void UKF::PredictLidarMeasurement(MatrixXd* Zsig_pred_out, VectorXd* z_pred_out, MatrixXd* S_pred_out) {

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(this->n_lasz_, 2 * this->n_aug_ + 1);
  Zsig.fill(0.0);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * this->n_aug_ + 1; i++) {

    // extract values for better readibility
    double p_x = this->Xsig_pred_(0,i);
    double p_y = this->Xsig_pred_(1,i);
    double v   = this->Xsig_pred_(2,i);
    double yaw = this->Xsig_pred_(3,i);

    // measurement model
    Zsig(0,i) = p_x;
    Zsig(1,i) = p_y;
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(this->n_lasz_);
  z_pred.fill(0.0);
  for (int i=0; i < 2 * this->n_aug_ + 1; i++) {
      z_pred = z_pred + this->weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(this->n_lasz_,this->n_lasz_);
  S.fill(0.0);
  for (int i = 0; i < 2 * this->n_aug_ + 1; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + this->weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(this->n_lasz_,this->n_lasz_);
  R(0,0) = this->std_laspx_ * this->std_laspx_;
  R(1,1) = this->std_laspy_ * this->std_laspy_;
  S = S + R;

  //write result
  *Zsig_pred_out = Zsig;
  *z_pred_out = z_pred;
  *S_pred_out = S;
}

void UKF::PredictRadarMeasurement(MatrixXd* Zsig_pred_out, VectorXd* z_pred_out, MatrixXd* S_pred_out) {

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(this->n_radz_, 2 * this->n_aug_ + 1);
  Zsig.fill(0.0);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * this->n_aug_ + 1; i++) {

    // extract values for better readibility
    double p_x = this->Xsig_pred_(0,i);
    double p_y = this->Xsig_pred_(1,i);
    double v   = this->Xsig_pred_(2,i);
    double yaw = this->Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y); //rho
    if(p_y > 0.0001 && p_x > 0.0001) {
      Zsig(1,i) = atan2(p_y,p_x); //phi
    }
    else {
      Zsig(1,i) = 0;
    }

    if(Zsig(0,i) > 0.0001) {
      Zsig(2,i) = (p_x*v1 + p_y*v2 ) / Zsig(0,i); //rho_dot
    }
    else {
      Zsig(2,i) = 0;
    }
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(this->n_radz_);
  z_pred.fill(0.0);
  for (int i=0; i < 2 * this->n_aug_ + 1; i++) {
      z_pred = z_pred + this->weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix Sz_diff
  MatrixXd S = MatrixXd(this->n_radz_,this->n_radz_);
  S.fill(0.0);
  for (int i = 0; i < 2 * this->n_aug_ + 1; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    z_diff(1) = NormalizeAngle(z_diff(1));

    S = S + this->weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(this->n_radz_,this->n_radz_);
  R(0,0) = this->std_radr_* this->std_radr_;
  R(1,1) = this->std_radphi_* this->std_radphi_;
  R(2,2) = this->std_radrd_ * this->std_radrd_;
  S = S + R;

  //write result
  *Zsig_pred_out = Zsig;
  *z_pred_out = z_pred;
  *S_pred_out = S;
}


void UKF::UpdateState(MatrixXd& Zsig_pred, VectorXd& z_pred, MatrixXd& S_pred, MeasurementPackage& meas_package) {


  //set measurement dimension (lidar = 2, radar = 3)
  int n_z = meas_package.raw_measurements_.size();

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(this->n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * this->n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig_pred.col(i) - z_pred;

    if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      //angle normalization
      z_diff(1) = NormalizeAngle(z_diff(1));
    }

    // state difference fixing
    VectorXd x_diff = this->Xsig_pred_.col(i) - this->x_;

    //angle normalization
    x_diff(3) = NormalizeAngle(x_diff(3));
    x_diff(4) = NormalizeAngle(x_diff(4));

    Tc = Tc + this->weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S_pred.inverse();

  //residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    //angle normalization
    z_diff(1) = NormalizeAngle(z_diff(1));
  }

  //update state mean and covariance matrix
  this->x_ = this->x_ + K * z_diff;
  this->P_ = this->P_ - K * S_pred * K.transpose();

}
