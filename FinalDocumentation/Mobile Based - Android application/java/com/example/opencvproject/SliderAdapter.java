package com.example.opencvproject;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import org.opencv.android.Utils;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.List;

public class SliderAdapter extends RecyclerView.Adapter<SliderAdapter.SliderViewHolder>{

    private List<Mat> tsList;
    private Context context;

    public SliderAdapter(List<Mat> tsList, Context context){
        this.tsList = tsList;
        this.context=context;
    }

    @NonNull
    @Override
    public SliderViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {

        return new SliderViewHolder(LayoutInflater.from(parent.getContext()).inflate(
                R.layout.slide_item_container,
                parent,
                false
        ));
    }

    @Override
    public void onBindViewHolder(@NonNull SliderViewHolder holder, int position) {
        holder.setImage(tsList.get(position));
    }

    @Override
    public int getItemCount() {
        return tsList.size();
    }

    class SliderViewHolder extends RecyclerView.ViewHolder{

        public SliderViewHolder(View itemView) {
            super(itemView);

        }

        void setImage(Mat ts){
            ImageView imageView= itemView.findViewById(R.id.imageSlide);
            imageView.setImageBitmap(convertMatToBitMap(ts));
        }

        Bitmap convertMatToBitMap(Mat input){
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
}
