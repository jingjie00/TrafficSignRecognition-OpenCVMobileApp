package com.example.opencvproject;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.os.Bundle;
import android.os.Handler;
import android.widget.Toast;

import com.tbruyelle.rxpermissions2.RxPermissions;


public class SplashScreen extends AppCompatActivity {

    @SuppressLint("CheckResult")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash_screen);
        getSupportActionBar().hide();
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        Handler handler=new Handler();
        RxPermissions rxPermissions = new RxPermissions(this);
        rxPermissions
                .request(Manifest.permission.CAMERA)
                .subscribe(granted -> {
                    if (granted) {
                        handler.postDelayed(new Runnable() {
                            @Override
                            public void run() {
                                Intent intent =new Intent(SplashScreen.this, MainActivity.class);
                                finish();
                                startActivity(intent);
                            }
                        },2000);
                        // All requested permissions are granted
                    } else {
                        Toast.makeText(this, "Permissions Denied. Please Accept the Needed Permissions. ", Toast.LENGTH_LONG).show();
                        // At least one permission is denied
                    }
                });



    }
}