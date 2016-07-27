/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

#include <algorithm>
#include <iterator>
#include <limits>

namespace cv
{

int RANSACUpdateNumIters( double p, double ep, int modelPoints, int maxIters )
{
    if( modelPoints <= 0 )
        CV_Error( Error::StsOutOfRange, "the number of model points should be positive" );

    p = MAX(p, 0.);
    p = MIN(p, 1.);
    ep = MAX(ep, 0.);
    ep = MIN(ep, 1.);

    // avoid inf's & nan's
    double num = MAX(1. - p, DBL_MIN);
    double denom = 1. - std::pow(1. - ep, modelPoints);
    if( denom < DBL_MIN )
        return 0;

    num = std::log(num);
    denom = std::log(denom);

    return denom >= 0 || -num >= maxIters*(-denom) ? maxIters : cvRound(num/denom);
}


class RANSACPointSetRegistrator : public PointSetRegistrator
{
public:
    RANSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb=Ptr<PointSetRegistrator::Callback>(),
                              int _modelPoints=0, double _threshold=0, double _confidence=0.99, int _maxIters=1000)
    : cb(_cb), modelPoints(_modelPoints), threshold(_threshold), confidence(_confidence), maxIters(_maxIters)
    {
        checkPartialSubsets = false;
    }

    int findInliers( const Mat& m1, const Mat& m2, const Mat& model, Mat& err, Mat& mask, double thresh ) const
    {
        cb->computeError( m1, m2, model, err );
        mask.create(err.size(), CV_8U);

        CV_Assert( err.isContinuous() && err.type() == CV_32F && mask.isContinuous() && mask.type() == CV_8U);
        const float* errptr = err.ptr<float>();
        uchar* maskptr = mask.ptr<uchar>();
        float t = (float)(thresh*thresh);
        int i, n = (int)err.total(), nz = 0;
        for( i = 0; i < n; i++ )
        {
            int f = errptr[i] <= t;
            maskptr[i] = (uchar)f;
            nz += f;
        }
        return nz;
    }

    bool getSubset( const Mat& m1, const Mat& m2,
                    Mat& ms1, Mat& ms2, RNG& rng,
                    int maxAttempts=1000 ) const
    {
        cv::AutoBuffer<int> _idx(modelPoints);
        int* idx = _idx;
        int i = 0, j, k, iters = 0;
        int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
        int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
        int esz1 = (int)m1.elemSize1()*d1, esz2 = (int)m2.elemSize1()*d2;
        int count = m1.checkVector(d1), count2 = m2.checkVector(d2);
        const int *m1ptr = m1.ptr<int>(), *m2ptr = m2.ptr<int>();

        ms1.create(modelPoints, 1, CV_MAKETYPE(m1.depth(), d1));
        ms2.create(modelPoints, 1, CV_MAKETYPE(m2.depth(), d2));

        int *ms1ptr = ms1.ptr<int>(), *ms2ptr = ms2.ptr<int>();

        CV_Assert( count >= modelPoints && count == count2 );
        CV_Assert( (esz1 % sizeof(int)) == 0 && (esz2 % sizeof(int)) == 0 );
        esz1 /= sizeof(int);
        esz2 /= sizeof(int);

        for(; iters < maxAttempts; iters++)
        {
            for( i = 0; i < modelPoints && iters < maxAttempts; )
            {
                int idx_i = 0;
                for(;;)
                {
                    idx_i = idx[i] = rng.uniform(0, count);
                    for( j = 0; j < i; j++ )
                        if( idx_i == idx[j] )
                            break;
                    if( j == i )
                        break;
                }
                for( k = 0; k < esz1; k++ )
                    ms1ptr[i*esz1 + k] = m1ptr[idx_i*esz1 + k];
                for( k = 0; k < esz2; k++ )
                    ms2ptr[i*esz2 + k] = m2ptr[idx_i*esz2 + k];
                if( checkPartialSubsets && !cb->checkSubset( ms1, ms2, i+1 ))
                {
                    // we may have selected some bad points;
                    // so, let's remove some of them randomly
                    i = rng.uniform(0, i+1);
                    iters++;
                    continue;
                }
                i++;
            }
            if( !checkPartialSubsets && i == modelPoints && !cb->checkSubset(ms1, ms2, i))
                continue;
            break;
        }

        return i == modelPoints && iters < maxAttempts;
    }

    bool run(InputArray _m1, InputArray _m2, OutputArray _model, OutputArray _mask) const
    {
        bool result = false;
        Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        Mat err, mask, model, bestModel, ms1, ms2;

        int iter, niters = MAX(maxIters, 1);
        int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
        int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
        int count = m1.checkVector(d1), count2 = m2.checkVector(d2), maxGoodCount = 0;

        RNG rng((uint64)-1);

        CV_Assert( cb );
        CV_Assert( confidence > 0 && confidence < 1 );

        CV_Assert( count >= 0 && count2 == count );
        if( count < modelPoints )
            return false;

        Mat bestMask0, bestMask;

        if( _mask.needed() )
        {
            _mask.create(count, 1, CV_8U, -1, true);
            bestMask0 = bestMask = _mask.getMat();
            CV_Assert( (bestMask.cols == 1 || bestMask.rows == 1) && (int)bestMask.total() == count );
        }
        else
        {
            bestMask.create(count, 1, CV_8U);
            bestMask0 = bestMask;
        }

        if( count == modelPoints )
        {
            if( cb->runKernel(m1, m2, bestModel) <= 0 )
                return false;
            bestModel.copyTo(_model);
            bestMask.setTo(Scalar::all(1));
            return true;
        }

        for( iter = 0; iter < niters; iter++ )
        {
            int i, nmodels;
            if( count > modelPoints )
            {
                bool found = getSubset( m1, m2, ms1, ms2, rng, 10000 );
                if( !found )
                {
                    if( iter == 0 )
                        return false;
                    break;
                }
            }

            nmodels = cb->runKernel( ms1, ms2, model );
            if( nmodels <= 0 )
                continue;
            CV_Assert( model.rows % nmodels == 0 );
            Size modelSize(model.cols, model.rows/nmodels);

            for( i = 0; i < nmodels; i++ )
            {
                Mat model_i = model.rowRange( i*modelSize.height, (i+1)*modelSize.height );
                int goodCount = findInliers( m1, m2, model_i, err, mask, threshold );

                if( goodCount > MAX(maxGoodCount, modelPoints-1) )
                {
                    std::swap(mask, bestMask);
                    model_i.copyTo(bestModel);
                    maxGoodCount = goodCount;
                    niters = RANSACUpdateNumIters( confidence, (double)(count - goodCount)/count, modelPoints, niters );
                }
            }
        }

        if( maxGoodCount > 0 )
        {
            if( bestMask.data != bestMask0.data )
            {
                if( bestMask.size() == bestMask0.size() )
                    bestMask.copyTo(bestMask0);
                else
                    transpose(bestMask, bestMask0);
            }
            bestModel.copyTo(_model);
            result = true;
        }
        else
            _model.release();

        return result;
    }

    void setCallback(const Ptr<PointSetRegistrator::Callback>& _cb) { cb = _cb; }

    Ptr<PointSetRegistrator::Callback> cb;
    int modelPoints;
    bool checkPartialSubsets;
    double threshold;
    double confidence;
    int maxIters;
};

class LMeDSPointSetRegistrator : public RANSACPointSetRegistrator
{
public:
    LMeDSPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb=Ptr<PointSetRegistrator::Callback>(),
                              int _modelPoints=0, double _confidence=0.99, int _maxIters=1000)
    : RANSACPointSetRegistrator(_cb, _modelPoints, 0, _confidence, _maxIters) {}

    bool run(InputArray _m1, InputArray _m2, OutputArray _model, OutputArray _mask) const
    {
        const double outlierRatio = 0.45;
        bool result = false;
        Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        Mat ms1, ms2, err, errf, model, bestModel, mask, mask0;

        int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
        int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
        int count = m1.checkVector(d1), count2 = m2.checkVector(d2);
        double minMedian = DBL_MAX;

        RNG rng((uint64)-1);

        CV_Assert( cb );
        CV_Assert( confidence > 0 && confidence < 1 );

        CV_Assert( count >= 0 && count2 == count );
        if( count < modelPoints )
            return false;

        if( _mask.needed() )
        {
            _mask.create(count, 1, CV_8U, -1, true);
            mask0 = mask = _mask.getMat();
            CV_Assert( (mask.cols == 1 || mask.rows == 1) && (int)mask.total() == count );
        }

        if( count == modelPoints )
        {
            if( cb->runKernel(m1, m2, bestModel) <= 0 )
                return false;
            bestModel.copyTo(_model);
            mask.setTo(Scalar::all(1));
            return true;
        }

        int iter, niters = RANSACUpdateNumIters(confidence, outlierRatio, modelPoints, maxIters);
        niters = MAX(niters, 3);

        for( iter = 0; iter < niters; iter++ )
        {
            int i, nmodels;
            if( count > modelPoints )
            {
                bool found = getSubset( m1, m2, ms1, ms2, rng );
                if( !found )
                {
                    if( iter == 0 )
                        return false;
                    break;
                }
            }

            nmodels = cb->runKernel( ms1, ms2, model );
            if( nmodels <= 0 )
                continue;

            CV_Assert( model.rows % nmodels == 0 );
            Size modelSize(model.cols, model.rows/nmodels);

            for( i = 0; i < nmodels; i++ )
            {
                Mat model_i = model.rowRange( i*modelSize.height, (i+1)*modelSize.height );
                cb->computeError( m1, m2, model_i, err );
                if( err.depth() != CV_32F )
                    err.convertTo(errf, CV_32F);
                else
                    errf = err;
                CV_Assert( errf.isContinuous() && errf.type() == CV_32F && (int)errf.total() == count );
                std::nth_element(errf.ptr<int>(), errf.ptr<int>() + count/2, errf.ptr<int>() + count);
                double median = errf.at<float>(count/2);

                if( median < minMedian )
                {
                    minMedian = median;
                    model_i.copyTo(bestModel);
                }
            }
        }

        if( minMedian < DBL_MAX )
        {
            double sigma = 2.5*1.4826*(1 + 5./(count - modelPoints))*std::sqrt(minMedian);
            sigma = MAX( sigma, 0.001 );

            count = findInliers( m1, m2, bestModel, err, mask, sigma );
            if( _mask.needed() && mask0.data != mask.data )
            {
                if( mask0.size() == mask.size() )
                    mask.copyTo(mask0);
                else
                    transpose(mask, mask0);
            }
            bestModel.copyTo(_model);
            result = count >= modelPoints;
        }
        else
            _model.release();

        return result;
    }

};

Ptr<PointSetRegistrator> createRANSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb,
                                                         int _modelPoints, double _threshold,
                                                         double _confidence, int _maxIters)
{
    return Ptr<PointSetRegistrator>(
        new RANSACPointSetRegistrator(_cb, _modelPoints, _threshold, _confidence, _maxIters));
}


Ptr<PointSetRegistrator> createLMeDSPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb,
                             int _modelPoints, double _confidence, int _maxIters)
{
    return Ptr<PointSetRegistrator>(
        new LMeDSPointSetRegistrator(_cb, _modelPoints, _confidence, _maxIters));
}


class Affine3DEstimatorCallback : public PointSetRegistrator::Callback
{
public:
    int runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const
    {
        Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        const Point3f* from = m1.ptr<Point3f>();
        const Point3f* to   = m2.ptr<Point3f>();

        const int N = 12;
        double buf[N*N + N + N];
        Mat A(N, N, CV_64F, &buf[0]);
        Mat B(N, 1, CV_64F, &buf[0] + N*N);
        Mat X(N, 1, CV_64F, &buf[0] + N*N + N);
        double* Adata = A.ptr<double>();
        double* Bdata = B.ptr<double>();
        A = Scalar::all(0);

        for( int i = 0; i < (N/3); i++ )
        {
            Bdata[i*3] = to[i].x;
            Bdata[i*3+1] = to[i].y;
            Bdata[i*3+2] = to[i].z;

            double *aptr = Adata + i*3*N;
            for(int k = 0; k < 3; ++k)
            {
                aptr[0] = from[i].x;
                aptr[1] = from[i].y;
                aptr[2] = from[i].z;
                aptr[3] = 1.0;
                aptr += 16;
            }
        }

        solve(A, B, X, DECOMP_SVD);
        X.reshape(1, 3).copyTo(_model);

        return 1;
    }

    void computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const
    {
        Mat m1 = _m1.getMat(), m2 = _m2.getMat(), model = _model.getMat();
        const Point3f* from = m1.ptr<Point3f>();
        const Point3f* to   = m2.ptr<Point3f>();
        const double* F = model.ptr<double>();

        int count = m1.checkVector(3);
        CV_Assert( count > 0 );

        _err.create(count, 1, CV_32F);
        Mat err = _err.getMat();
        float* errptr = err.ptr<float>();

        for(int i = 0; i < count; i++ )
        {
            const Point3f& f = from[i];
            const Point3f& t = to[i];

            double a = F[0]*f.x + F[1]*f.y + F[ 2]*f.z + F[ 3] - t.x;
            double b = F[4]*f.x + F[5]*f.y + F[ 6]*f.z + F[ 7] - t.y;
            double c = F[8]*f.x + F[9]*f.y + F[10]*f.z + F[11] - t.z;

            errptr[i] = (float)(a*a + b*b + c*c);
        }
    }

    bool checkSubset( InputArray _ms1, InputArray _ms2, int count ) const
    {
        const float threshold = 0.996f;
        Mat ms1 = _ms1.getMat(), ms2 = _ms2.getMat();

        for( int inp = 1; inp <= 2; inp++ )
        {
            int j, k, i = count - 1;
            const Mat* msi = inp == 1 ? &ms1 : &ms2;
            const Point3f* ptr = msi->ptr<Point3f>();

            CV_Assert( count <= msi->rows );

            // check that the i-th selected point does not belong
            // to a line connecting some previously selected points
            for(j = 0; j < i; ++j)
            {
                Point3f d1 = ptr[j] - ptr[i];
                float n1 = d1.x*d1.x + d1.y*d1.y;

                for(k = 0; k < j; ++k)
                {
                    Point3f d2 = ptr[k] - ptr[i];
                    float denom = (d2.x*d2.x + d2.y*d2.y)*n1;
                    float num = d1.x*d2.x + d1.y*d2.y;

                    if( num*num > threshold*threshold*denom )
                        return false;
                }
            }
        }
        return true;
    }
};

class Affine2DEstimatorCallback : public PointSetRegistrator::Callback
{
public:
    int runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const
    {
        Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        const Point2f* from = m1.ptr<Point2f>();
        const Point2f* to   = m2.ptr<Point2f>();

        // we need 3 points to estimate affine transform
        const int N = 6;
        double buf[N*N + N + N];
        Mat A(N, N, CV_64F, &buf[0]);
        Mat B(N, 1, CV_64F, &buf[0] + N*N);
        Mat X(N, 1, CV_64F, &buf[0] + N*N + N);
        double* Adata = A.ptr<double>();
        double* Bdata = B.ptr<double>();
        A = Scalar::all(0);

        for( int i = 0; i < (N/2); i++ )
        {
            Bdata[i*2] = to[i].x;
            Bdata[i*2+1] = to[i].y;

            double *aptr = Adata + i*2*N;
            for(int k = 0; k < 2; ++k)
            {
                aptr[0] = from[i].x;
                aptr[1] = from[i].y;
                aptr[2] = 1.0;
                aptr += N+3;
            }
        }

        solve(A, B, X, DECOMP_SVD);
        X.reshape(1, 2).copyTo(_model);
        return 1;
    }

    void computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const
    {
        Mat m1 = _m1.getMat(), m2 = _m2.getMat(), model = _model.getMat();
        const Point2f* from = m1.ptr<Point2f>();
        const Point2f* to   = m2.ptr<Point2f>();
        const double* F = model.ptr<double>();

        int count = m1.checkVector(2);
        CV_Assert( count > 0 );

        _err.create(count, 1, CV_32F);
        Mat err = _err.getMat();
        float* errptr = err.ptr<float>();

        for(int i = 0; i < count; i++ )
        {
            const Point2f& f = from[i];
            const Point2f& t = to[i];

            double a = F[0]*f.x + F[1]*f.y + F[2] - t.x;
            double b = F[3]*f.x + F[4]*f.y + F[5] - t.y;

            errptr[i] = (float)(a*a + b*b);
        }
    }

    bool checkSubset( InputArray _ms1, InputArray _ms2, int count ) const
    {
        const float threshold = 0.996f;
        Mat ms1 = _ms1.getMat(), ms2 = _ms2.getMat();

        for( int inp = 1; inp <= 2; inp++ )
        {
            int j, k, i = count - 1;
            const Mat* msi = inp == 1 ? &ms1 : &ms2;
            const Point2f* ptr = msi->ptr<Point2f>();

            CV_Assert( count <= msi->rows );

            // check that the i-th selected point does not belong
            // to a line connecting some previously selected points
            for(j = 0; j < i; ++j)
            {
                Point2f d1 = ptr[j] - ptr[i];
                float n1 = d1.x*d1.x + d1.y*d1.y;

                for(k = 0; k < j; ++k)
                {
                    Point2f d2 = ptr[k] - ptr[i];
                    float denom = (d2.x*d2.x + d2.y*d2.y)*n1;
                    float num = d1.x*d2.x + d1.y*d2.y;

                    if( num*num > threshold*threshold*denom )
                        return false;
                }
            }
        }
        return true;
    }
};

class AffinePartial2DEstimatorCallback : public Affine2DEstimatorCallback
{
public:
    int runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const
    {
        Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        const Point2f* from = m1.ptr<Point2f>();
        const Point2f* to   = m2.ptr<Point2f>();

        // for partial affine trasform we need only 2 points
        const int N = 4;
        double buf[N*N + N + N + 2*3];
        Mat A(N, N, CV_64F, buf);
        Mat B(N, 1, CV_64F, buf + N*N);
        Mat X(N, 1, CV_64F, buf + N*N + N);
        Mat M(2, 3, CV_64F, buf + N*N + N + N);
        double* Adata = A.ptr<double>();
        double* Bdata = B.ptr<double>();
        double* Xdata = X.ptr<double>();
        double* Mdata = M.ptr<double>();
        A = Scalar::all(0);

        for( int i = 0; i < (N/2); i++ )
        {
            Bdata[i*2] = to[i].x;
            Bdata[i*2+1] = to[i].y;

            double *aptr = Adata + i*2*N;
            aptr[0] = from[i].x;
            aptr[1] = -from[i].y;
            aptr[2] = 1.0;
            aptr[4] = from[i].y;
            aptr[5] = from[i].x;
            aptr[7] = 1.0;
        }

        solve(A, B, X, DECOMP_SVD);

        // set model, rotation part is antisymmetric
        Mdata[0] = Mdata[4] = Xdata[0];
        Mdata[1] = -Xdata[1];
        Mdata[2] = Xdata[2];
        Mdata[3] = Xdata[1];
        Mdata[5] = Xdata[3];
        M.copyTo(_model);
        return 1;
    }
};

class Affine2DRefineCallback : public LMSolver::Callback
{
public:
    Affine2DRefineCallback(InputArray _src, InputArray _dst)
    {
        src = _src.getMat();
        dst = _dst.getMat();
    }

    bool compute(InputArray _param, OutputArray _err, OutputArray _Jac) const
    {
        int i, count = src.checkVector(2);
        Mat param = _param.getMat();
        _err.create(count*2, 1, CV_64F);
        Mat err = _err.getMat(), J;
        if( _Jac.needed())
        {
            _Jac.create(count*2, param.rows, CV_64F);
            J = _Jac.getMat();
            CV_Assert( J.isContinuous() && J.cols == 6 );
        }

        const Point2f* M = src.ptr<Point2f>();
        const Point2f* m = dst.ptr<Point2f>();
        const double* h = param.ptr<double>();
        double* errptr = err.ptr<double>();
        double* Jptr = J.data ? J.ptr<double>() : 0;

        for( i = 0; i < count; i++ )
        {
            double Mx = M[i].x, My = M[i].y;
            double xi = h[0]*Mx + h[1]*My + h[2];
            double yi = h[3]*Mx + h[4]*My + h[5];
            errptr[i*2] = xi - m[i].x;
            errptr[i*2+1] = yi - m[i].y;

            /*
            Jacobian should be:
                {x, y, 1, 0, 0, 0}
                {0, 0, 0, x, y, 1}
            */
            if( Jptr )
            {
                Jptr[0] = Mx; Jptr[1] = My; Jptr[2] = 1.;
                Jptr[3] = Jptr[4] = Jptr[5] = 0.;
                Jptr[6] = Jptr[7] = Jptr[8] = 0.;
                Jptr[9] = Mx; Jptr[10] = My; Jptr[11] = 1.;

                Jptr += 6*2;
            }
        }

        return true;
    }

    Mat src, dst;
};

class AffinePartial2DRefineCallback : public LMSolver::Callback
{
public:
    AffinePartial2DRefineCallback(InputArray _src, InputArray _dst)
    {
        src = _src.getMat();
        dst = _dst.getMat();
    }

    bool compute(InputArray _param, OutputArray _err, OutputArray _Jac) const
    {
        int i, count = src.checkVector(2);
        Mat param = _param.getMat();
        _err.create(count*2, 1, CV_64F);
        Mat err = _err.getMat(), J;
        if( _Jac.needed())
        {
            _Jac.create(count*2, param.rows, CV_64F);
            J = _Jac.getMat();
            CV_Assert( J.isContinuous() && J.cols == 4 );
        }

        const Point2f* M = src.ptr<Point2f>();
        const Point2f* m = dst.ptr<Point2f>();
        const double* h = param.ptr<double>();
        double* errptr = err.ptr<double>();
        double* Jptr = J.data ? J.ptr<double>() : 0;

        for( i = 0; i < count; i++ )
        {
            double Mx = M[i].x, My = M[i].y;
            double xi = h[0]*Mx - h[1]*My + h[2];
            double yi = h[1]*Mx + h[0]*My + h[3];
            errptr[i*2] = xi - m[i].x;
            errptr[i*2+1] = yi - m[i].y;

            /*
            Jacobian should be:
                {x, -y, 1, 0}
                {y,  x, 0, 1}
            */
            if( Jptr )
            {
                Jptr[0] = Mx; Jptr[1] = -My; Jptr[2] = 1.; Jptr[3] = 0.;
                Jptr[4] = My; Jptr[5] =  Mx; Jptr[6] = 0.; Jptr[7] = 1.;

                Jptr += 4*2;
            }
        }

        return true;
    }

    Mat src, dst;
};

}

int cv::estimateAffine3D(InputArray _from, InputArray _to,
                         OutputArray _out, OutputArray _inliers,
                         double param1, double param2)
{
    CV_INSTRUMENT_REGION()

    Mat from = _from.getMat(), to = _to.getMat();
    int count = from.checkVector(3);

    CV_Assert( count >= 0 && to.checkVector(3) == count );

    Mat dFrom, dTo;
    from.convertTo(dFrom, CV_32F);
    to.convertTo(dTo, CV_32F);
    dFrom = dFrom.reshape(3, count);
    dTo = dTo.reshape(3, count);

    const double epsilon = DBL_EPSILON;
    param1 = param1 <= 0 ? 3 : param1;
    param2 = (param2 < epsilon) ? 0.99 : (param2 > 1 - epsilon) ? 0.99 : param2;

    return createRANSACPointSetRegistrator(makePtr<Affine3DEstimatorCallback>(), 4, param1, param2)->run(dFrom, dTo, _out, _inliers);
}

int cv::estimateAffine2D(InputArray _from, InputArray _to,
                         OutputArray _H, OutputArray _inliers,
                         double param1, double param2)
{
    Mat from = _from.getMat(), to = _to.getMat();
    int count = from.checkVector(2);

    CV_Assert( count >= 0 && to.checkVector(2) == count );

    Mat dFrom, dTo, H, tempMask;

    from.convertTo(dFrom, CV_32F);
    to.convertTo(dTo, CV_32F);
    dFrom = dFrom.reshape(2, count);
    dTo = dTo.reshape(2, count);

    const double epsilon = DBL_EPSILON;
    param1 = param1 <= 0 ? 3 : param1;
    param2 = (param2 < epsilon) ? 0.99 : (param2 > 1 - epsilon) ? 0.99 : param2;

    // run RANSAC
    bool result = false;
    Ptr<PointSetRegistrator::Callback> cb = makePtr<Affine2DEstimatorCallback>();
    result = createRANSACPointSetRegistrator(cb, 3, param1, param2)->run(dFrom, dTo, H, tempMask);

    if(result && count > 3)
    {
        // reorder to start with inliers
        compressElems(dFrom.ptr<Point2f>(), tempMask.ptr<uchar>(), 1, count);
        int inliers_count = compressElems(dTo.ptr<Point2f>(), tempMask.ptr<uchar>(), 1, count);
        if(inliers_count > 0)
        {
            Mat src = dFrom.rowRange(0, inliers_count);
            Mat dst = dTo.rowRange(0, inliers_count);
            Mat Hvec = H.reshape(1, 6);
            createLMSolver(makePtr<Affine2DRefineCallback>(src, dst), 10)->run(Hvec);
        }
    }

    if(result)
    {
        if(_inliers.needed())
            tempMask.copyTo(_inliers);
        if(_H.needed())
            H.copyTo(_H);
    }
    else
    {
        if(_inliers.needed())
        {
            tempMask = Mat::zeros(count, 1, CV_8U);
            tempMask.copyTo(_inliers);
        }
    }

    return result;
}

int cv::estimateAffinePartial2D(InputArray _from, InputArray _to,
                         OutputArray _H, OutputArray _inliers,
                         double param1, double param2)
{
    Mat from = _from.getMat(), to = _to.getMat();
    const int count = from.checkVector(2);

    CV_Assert( count >= 0 && to.checkVector(2) == count );

    Mat dFrom, dTo, H, tempMask;

    from.convertTo(dFrom, CV_32F);
    to.convertTo(dTo, CV_32F);
    dFrom = dFrom.reshape(2, count);
    dTo = dTo.reshape(2, count);

    const double epsilon = DBL_EPSILON;
    param1 = param1 <= 0 ? 3 : param1;
    param2 = (param2 < epsilon) ? 0.99 : (param2 > 1 - epsilon) ? 0.99 : param2;

    // run RANSAC
    bool result = false;
    Ptr<PointSetRegistrator::Callback> cb = makePtr<AffinePartial2DEstimatorCallback>();
    result = createRANSACPointSetRegistrator(cb, 2, param1, param2)->run(dFrom, dTo, H, tempMask);

    if(result && count > 2)
    {
        // reorder to start with inliers
        compressElems(dFrom.ptr<Point2f>(), tempMask.ptr<uchar>(), 1, count);
        int inliers_count = compressElems(dTo.ptr<Point2f>(), tempMask.ptr<uchar>(), 1, count);
        if(inliers_count > 0)
        {
            Mat src = dFrom.rowRange(0, inliers_count);
            Mat dst = dTo.rowRange(0, inliers_count);
            // H is
            //     a -b tx
            //     b  a ty
            // Hvec model for LevMarq is
            //     (a, b, tx, ty)
            double *Hptr = H.ptr<double>();
            double Hvec_buf[4] = {Hptr[0], Hptr[3], Hptr[2], Hptr[5]};
            Mat Hvec (4, 1, CV_64F, Hvec_buf);
            createLMSolver(makePtr<AffinePartial2DRefineCallback>(src, dst), 10)->run(Hvec);
            // update H with refined parameters
            Hptr[0] = Hptr[4] = Hvec_buf[0];
            Hptr[1] = -Hvec_buf[1];
            Hptr[2] = Hvec_buf[2];
            Hptr[3] = Hvec_buf[1];
            Hptr[5] = Hvec_buf[3];
        }
    }

    if(result)
    {
        if(_inliers.needed())
            tempMask.copyTo(_inliers);
        if(_H.needed())
            H.copyTo(_H);
    }
    else
    {
        if(_inliers.needed())
        {
            tempMask = Mat::zeros(count, 1, CV_8U);
            tempMask.copyTo(_inliers);
        }
    }

    return result;
}
