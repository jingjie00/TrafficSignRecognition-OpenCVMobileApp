package com.example.opencvproject;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.view.SurfaceView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import static org.opencv.imgproc.Imgproc.contourArea;
import static org.opencv.imgproc.Imgproc.dilate;
import static org.opencv.imgproc.Imgproc.drawContours;
import static org.opencv.imgproc.Imgproc.erode;
import static org.opencv.imgproc.Imgproc.findContours;
import static org.opencv.imgproc.Imgproc.floodFill;

import com.google.android.material.floatingactionbutton.FloatingActionButton;

public class FromCameraRecognitionActivity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "FromCamera";
    int count = 0;

    TextView textView;
    Button capture;
    private Mat mRgba;
    private GtsrbClassifier gtsrbClassifier;
    private CameraBridgeViewBase mOpenCvCameraView;
    Button navigate_from_file;
    List<Mat> result= new ArrayList<>();
    TextToSpeech tts;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public FromCameraRecognitionActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(R.layout.activity_from_camera_recognition);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        loadGtsrbClassifier();

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.cameraview);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);



        capture = findViewById(R.id.take_photo_btn);
        textView = findViewById(R.id.textView);

        textView.setText("Hello");
        capture.bringToFront();

        tts=new TextToSpeech(FromCameraRecognitionActivity.this, new TextToSpeech.OnInitListener() {

            @Override
            public void onInit(int status) {
                // TODO Auto-generated method stub
                if(status == TextToSpeech.SUCCESS){
                    int result=tts.setLanguage(Locale.US);
                    if(result==TextToSpeech.LANG_MISSING_DATA ||
                            result==TextToSpeech.LANG_NOT_SUPPORTED){
                        Log.e("error", "This Language is not supported");
                    }
                    else{
                        tts.setLanguage(Locale.US);
                    }
                }
                else
                    Log.e("error", "Initilization Failed!");
            }


        });


        //voice output
        //export
        navigate_from_file = findViewById(R.id.navigate_from_file);
        navigate_from_file.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Toast.makeText(getApplicationContext(), "Add face activity", Toast.LENGTH_SHORT).show();
                Intent intent = new Intent(FromCameraRecognitionActivity.this, FromFileRecognitionActivity.class);
                startActivity(intent);
                finish();
            }
        });

        capture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (result.size() == 0)
                    return;
                Mat currentCapture = result.get(0);
                Log.d("tmp", currentCapture.height() + " | " + currentCapture.width());
                try {
                    Bitmap bmp = convertMatToBitMap(currentCapture);
                    //imageView.setImageBitmap( convertMatToBitMap(currentCapture));
                    //imageView.invalidate();
                    Log.d("Exception", "hre");
                    Bitmap squareBitmap = ThumbnailUtils.extractThumbnail(bmp, bmp.getWidth(),bmp.getHeight());
                    Bitmap preprocessedImage = ImageUtils.prepareImageForClassification(squareBitmap);
                    List<Classification> recognitions = gtsrbClassifier.recognizeImage(preprocessedImage);
                    String a="";
                    for (Classification b:recognitions)
                        a=a+b;
                    textView.setText(a);
                    textView.invalidate();
                    tts.speak(textView.getText(),TextToSpeech.QUEUE_ADD,null,"0");
                } catch (Exception e) {
                    Log.d("Exception", ""+e.getMessage());
                }
            }
        });

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

    private void loadGtsrbClassifier() {
        try {
            gtsrbClassifier = new GtsrbClassifier(new Interpreter(loadModelFile(this, "gtsrb_model.tflite")));
        } catch (IOException e) {
            Toast.makeText(this, "GTSRB model couldn't be loaded. Check logs for details.", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
    }

    public MappedByteBuffer loadModelFile(Activity activity, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
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

        Mat input = inputFrame.rgba();
        ImagePreprocess ip = new ImagePreprocess();
        Thread thread = new Thread() {
            public void run() {
                result = ip.processCamera(input);
                mRgba = input;
            }
        };
        thread.run();
        return mRgba;
    }

}