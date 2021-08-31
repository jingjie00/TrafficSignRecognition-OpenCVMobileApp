package com.example.opencvproject;

import android.graphics.Bitmap;
import android.util.Log;
import android.widget.TextView;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import static org.opencv.core.CvType.CV_8UC3;

public class ImagePreprocess{
    private Mat input;

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


    ImagePreprocess(){
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


    public List<Mat> process(Mat in) {

        input=in;
        Mat hsv = new Mat();

        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_BGR2HSV);

        Mat		Image= new Mat(),
                redMask1= new Mat(), redMask2= new Mat(), blueMask= new Mat(), yellowMask= new Mat(), mask= new Mat(),
                canvas= new Mat();

        List<Mat> result =new ArrayList<Mat>();

        canvas.create(input.rows(), input.cols(), CV_8UC3);
        input.copyTo(canvas);

        // Match for Red
        Core.inRange(hsv, redLow1, redHigh1, redMask1);
        Core.inRange(hsv, redLow2, redHigh2, redMask2);

        // Match for Blue
        Core.inRange(hsv, blueLow, blueHigh, blueMask);

        // Match for Yellow
        Core.inRange(hsv, yellowLow, yellowHigh, yellowMask);

        //Remove noise
        Mat element = Imgproc.getStructuringElement(2, new Size(2*2+1, 2*2+1), new Point(2, 2) );
        Imgproc.morphologyEx(yellowMask,yellowMask, Imgproc.MORPH_CLOSE, element);
        Imgproc.morphologyEx(redMask1,redMask1, Imgproc.MORPH_CLOSE, element);
        Imgproc.morphologyEx(redMask2,redMask2, Imgproc.MORPH_CLOSE, element);
        Imgproc.morphologyEx(blueMask,blueMask, Imgproc.MORPH_CLOSE, element);

        //Merged
        Core.bitwise_or(redMask1, redMask2, mask);
        Core.bitwise_or(mask, blueMask, mask);
        Core.bitwise_or(mask, yellowMask, mask);

        //Convert to do bitwise
        Imgproc.cvtColor( mask,mask, Imgproc.COLOR_GRAY2BGR);
        Imgproc.cvtColor( hsv,Image, Imgproc.COLOR_HSV2BGR);
        Core.bitwise_and(mask, Image , Image);



        // Do contour
        Imgproc.erode(Image,Image, element);
        Imgproc.dilate(Image,Image, element);
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
        List<MatOfPoint> filter = new ArrayList<>(contours.size());

        for (int i=0; i<contours.size();i++){
            if(Imgproc.contourArea(contours.get(i))*6>Imgproc.contourArea(contours.get(contours.size()-1))){
                filter.add(contours.get(i));
            }
        }
        Rect boundRect;
        for (int i=0;i< filter.size();i++){
            Mat filledCanvasMask=new Mat(input.rows(), input.cols(),CV_8UC3,new Scalar(0,0,0));
            Imgproc.fillConvexPoly(filledCanvasMask,filter.get(i), new Scalar(255,255,255));
            Mat segment=Image.clone();
            Core.bitwise_and(filledCanvasMask, segment , segment);
            boundRect = Imgproc.boundingRect((MatOfPoint) filter.get(i));
            Imgproc.rectangle( input, boundRect.tl(), boundRect.br(), new Scalar(255,255,255), 2, Imgproc.LINE_AA, 0 );
            Mat crop=segment.submat(boundRect);

            result.add(crop);
        }

        if(result.size()<=6)
            return result;
        return result.subList(result.size()-6,result.size()-1);
    }

    private Scalar converScalarHsv2Rgba(Scalar hsvColor) {
        Mat pointMatRgba = new Mat();
        Mat pointMatHsv = new Mat(1, 1, CV_8UC3, hsvColor);
        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);

        return new Scalar(pointMatRgba.get(0, 0));
    }

    private static Bitmap convertMatToBitMap(Mat input){
        Bitmap bmp = null;
        Mat rgb = new Mat();
        Imgproc.cvtColor(input, rgb, Imgproc.COLOR_BGR2RGB);
        Imgproc.cvtColor(rgb, rgb, Imgproc.COLOR_RGB2BGR);
        try {
            bmp = Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rgb, bmp);
        }
        catch (CvException e){
            Log.d("Exception",e.getMessage());
        }
        return bmp;
    }
}
