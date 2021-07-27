package com.example.opencvproject;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.os.Bundle;

import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.view.View.OnTouchListener;
import android.view.SurfaceView;
import android.widget.Toast;
import com.tbruyelle.rxpermissions2.RxPermissions;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    static{
        if(!OpenCVLoader.initDebug()){
            Log.d("TAG", "OpenCV not loaded");

        }

        else{
            Log.d("TAG", "OpenCV loaded");
        }
    }
    int iLowH= 101;
    int iHighH = 130;
    int iLowS = 50;
    int iHighS = 255;
    int iLowV=70;
    int iHighV = 255;

    Mat imgHSV, imgThresholded;
    JavaCameraView cameraView;


    Scalar sc1, sc2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        RxPermissions rxPermissions = new RxPermissions(this);
        rxPermissions
                .request(Manifest.permission.CAMERA)
                .subscribe(granted -> {
                    if (granted) {
                        // All requested permissions are granted
                    } else {
                        Toast.makeText(this, "Permissions Denied. Please Accept the Needed Permissions. ", Toast.LENGTH_LONG).show();
                        // At least one permission is denied
                    }
                });

        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        setContentView(R.layout.activity_main);

        sc1=new Scalar(iLowH, iLowS, iLowV);
        sc2 = new Scalar(iHighH, iHighS, iHighV);

        cameraView = (JavaCameraView) findViewById(R.id.cameraview);

        cameraView.setCameraIndex(0); //rear cam

        cameraView.setCvCameraViewListener(this);
        cameraView.enableView();



    }


    @Override
    public void onCameraViewStarted(int width, int height) {

        imgHSV = new Mat(width, height, CvType.CV_16UC4);
        imgThresholded = new Mat(width, height, CvType.CV_16UC4);
    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    protected void onPause() {
        super.onPause();
        cameraView.disableView();
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        Imgproc.cvtColor(inputFrame.rgba(), imgHSV, Imgproc.COLOR_BGR2HSV);


        Core.inRange(imgHSV, sc1, sc2, imgThresholded);
        return imgThresholded;
    }
}