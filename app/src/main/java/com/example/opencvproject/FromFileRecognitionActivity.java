package com.example.opencvproject;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.viewpager2.widget.ViewPager2;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.res.AssetFileDescriptor;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.DragEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.Utils;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;

public class FromFileRecognitionActivity extends AppCompatActivity{
    private static final int RESULT_LOAD_IMAGE = 1;
    Button fromfile;
    Button export;
    TextView textView;
    ImageView input;
    ViewPager2 viewPager2;
    private GtsrbClassifier gtsrbClassifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(R.layout.activity_from_file_recognition);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        getSupportActionBar().hide();

        loadGtsrbClassifier();

        fromfile=findViewById(R.id.from_file);
        export=findViewById(R.id.export);
        textView=findViewById(R.id.textView);
        input=findViewById(R.id.input);
        viewPager2 = findViewById(R.id.result);

        fromfile.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent photoPickerIntent = new Intent(Intent.ACTION_PICK);
                photoPickerIntent.setType("image/*");
                startActivityForResult(photoPickerIntent, RESULT_LOAD_IMAGE);
            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        Bitmap selectedImage;
        if (resultCode == RESULT_OK) {
            try {
                final Uri imageUri = data.getData();
                final InputStream imageStream = getContentResolver().openInputStream(imageUri);
                selectedImage = BitmapFactory.decodeStream(imageStream);
                input.setImageBitmap(selectedImage);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
                Toast.makeText(this, "Something went wrong", Toast.LENGTH_LONG).show();
                return;
            }
        }else {
            Toast.makeText(this, "You haven't picked Image",Toast.LENGTH_LONG).show();
            return;
        }

        Mat mat = new Mat();
        Bitmap bmp32 = selectedImage.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, mat);


        ImagePreprocess ip = new ImagePreprocess();
        List<Mat> segment= ip.process(mat);

        viewPager2.setAdapter(new SliderAdapter(segment,getApplicationContext()));

        viewPager2.setOnDragListener(new View.OnDragListener() {
            @Override
            public boolean onDrag(View v, DragEvent event) {
                int currentItem=viewPager2.getCurrentItem();
                Mat mat = segment.get(currentItem);
                Bitmap bmp=convertMatToBitMap(mat);
                Bitmap squareBitmap = ThumbnailUtils.extractThumbnail(bmp, bmp.getWidth(),bmp.getHeight());
                Bitmap preprocessedImage = ImageUtils.prepareImageForClassification(squareBitmap);
                List<Classification> recognitions = gtsrbClassifier.recognizeImage(preprocessedImage);
                textView.setText(currentItem+" : " + recognitions.toString());
                return false;
            }
        });



    }

    public void loadGtsrbClassifier() {
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


    static Bitmap convertMatToBitMap(Mat input){
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