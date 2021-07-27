package com.example.opencvproject;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Vector;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.content.pm.ActivityInfo;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.view.View.OnTouchListener;
import android.view.SurfaceView;

import static org.opencv.core.Core.inRange;
import static org.opencv.core.CvType.CV_8UC3;
import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_NONE;
import static org.opencv.imgproc.Imgproc.RETR_EXTERNAL;
import static org.opencv.imgproc.Imgproc.boundingRect;
import static org.opencv.imgproc.Imgproc.contourArea;
import static org.opencv.imgproc.Imgproc.dilate;
import static org.opencv.imgproc.Imgproc.drawContours;
import static org.opencv.imgproc.Imgproc.erode;
import static org.opencv.imgproc.Imgproc.findContours;
import static org.opencv.imgproc.Imgproc.floodFill;

public class MainActivity extends Activity implements  CvCameraViewListener2 {
    private static final String  TAG              = "MainActivity";

    private Mat                  mRgba;
    private boolean flag;

    Scalar redLow1;
    Scalar redHigh1;

    Scalar redLow2 ;
    Scalar redHigh2 ;

    // Hue Ranges for Blue
    Scalar blueLow;
    Scalar blueHigh ;

    // Hue Ranges for Yellow
    Scalar yellowLow ;
    Scalar yellowHigh ;

    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.cameraview);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        flag=true;

        redLow1 = new  Scalar(150, 140, 160);
        redHigh1 = new  Scalar(180, 255, 255);

        redLow2 = new  Scalar(0, 50, 50);
        redHigh2 = new  Scalar(3, 255, 255);

        // Hue Ranges for Blue
        blueLow = new  Scalar(100, 150, 100);
        blueHigh = new  Scalar(128, 255, 255);

        // Hue Ranges for Yellow
        yellowLow = new  Scalar(14, 100, 140);
        yellowHigh = new Scalar(30, 255, 255);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }




    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        if(!flag)
            return null;
        mRgba = inputFrame.rgba();
        Mat hsv = new Mat();
        Imgproc.cvtColor(inputFrame.rgba(), hsv, Imgproc.COLOR_BGR2HSV);

        Mat		Image= new Mat(),
                redMask1= new Mat(), redMask2= new Mat(), blueMask= new Mat(), yellowMask= new Mat(), mask= new Mat(),
                canvas= new Mat();

        canvas.create(mRgba.rows(), mRgba.cols(), CV_8UC3);
        mRgba.copyTo(canvas);

        // Match for Red
        Core.inRange(hsv, redLow1, redHigh1, redMask1);
        Core.inRange(hsv, redLow2, redHigh2, redMask2);

        // Match for Blue
        Core.inRange(hsv, blueLow, blueHigh, blueMask);

        // Match for Yellow
        Core.inRange(hsv, yellowLow, yellowHigh, yellowMask);

        //Merged
        Core.bitwise_or(redMask1, redMask2, mask);
        Core.bitwise_or(mask, blueMask, mask);
        Core.bitwise_or(mask, yellowMask, mask);

        //Convert to do bitwise
        Imgproc.cvtColor( mask,mask, Imgproc.COLOR_GRAY2BGR);
        Imgproc.cvtColor( hsv,Image, Imgproc.COLOR_HSV2BGR);
        Core.bitwise_and(mask, Image , Image);

        // Do contour
        Mat cannyOutput = new Mat();
        Imgproc.Canny(Image, cannyOutput, 100, 100 * 2);
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(cannyOutput, contours,hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        Collections.sort(contours, new Comparator<MatOfPoint>() {
            @Override
            public int compare(MatOfPoint o1, MatOfPoint o2) {
                long sumMop1 = 0;
                long sumMop2 = 0;
                for( Point p: o1.toList() ){
                    sumMop1 += p.x + p.y;
                }
                for( Point p: o2.toList() ){
                    sumMop2 += p.x + p.y;
                }
                if( sumMop1 > sumMop2)
                    return 1;
                else if( sumMop1 < sumMop2 )
                    return -1;
                else
                    return 0;
            }

        });

        //Set boundarys
        Rect boundRect;
        if(contours.size()>=1) {
            boundRect = Imgproc.boundingRect((MatOfPoint) contours.get(contours.size() - 1));
            Imgproc.rectangle( mRgba, boundRect.tl(), boundRect.br(), new Scalar(255,255,255), 2, Imgproc.LINE_AA, 0 );
        }

        flag=true;
        return mRgba;
    }


    private Scalar converScalarHsv2Rgba(Scalar hsvColor) {
        Mat pointMatRgba = new Mat();
        Mat pointMatHsv = new Mat(1, 1, CV_8UC3, hsvColor);
        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);

        return new Scalar(pointMatRgba.get(0, 0));
    }
}