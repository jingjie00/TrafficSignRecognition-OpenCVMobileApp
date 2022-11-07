# TrafficSignRecognitionOpenCV
You may press the clone button to your android IDE. Or you may download the zip code.

Or Install me using APK to your phone now!!
https://drive.google.com/drive/folders/1aaEw0QVMY54ajhuOp81mHzOFRQ7fZJhJ?usp=sharing
Only applicable for UTAR domain account

The developed mobile application can be run in Android Studio using the steps given below:

1. Go to https://sourceforge.net/projects/opencvlibrary/files/opencv-android/ and download the latest OpenCV Android library.

2. Create a new Android project using Android Studio 

3. Import the OpenCV module using by clicking on File -> New -> Import Module

4. Browse to the downloaded OpenCV Android library and select the java folder inside the sdk folder.

5. Browse to the OpenCV library module in the Project pane and open its build.gradle file.

6. Change the compiledSdkVersion and targetSdkVersion to the latest Android SDK version installed on the current device.
 
7. Navigate to File -> Project Structure -> app module -> Dependencies -> Module Dependency

8. Select the OpenCV library module

9. Add the native libraries (libs folder) into the Android project's app module main folder (ProjectName/app/src/main). Rename this folder to jniLibs. 

10. Add the following requirements into the AndroidManifest.xml file:

<uses-permission android:name="android.permission.CAMERA"/>
    <uses-feature android:name="android.hardware.camera" android:required="false"/>
    <uses-feature android:name="android.hardware.camera.autofocus" android:required="false"/>
    <uses-feature android:name="android.hardware.camera.front" android:required="false"/>
    <uses-feature android:name="android.hardware.camera.front.autofocus" android:required="false"/>

11. Copy the java(in "java" folder) and xml files(in "res" folder) from the folder provided by us and paste them in their respective directories in the project folder.

12. The folder and file are for reference purpose such as "build.gradle" and "AndroidManifest.xml"

13. Lastly, create a directory called ml and import the trained model(provided in "ml" folder)

14. Compile the program to APK and run in virtual machine or phone. The sample APK is provided which name: "mobileapplication.apk"

15. For easy setup just clone our github respository:
https://github.com/jjtan00/TrafficSignRecognitionOpenCV.git

16. Any issue please contact 

Tan Jing Jie with +6011-38100852 
or
Jacynth Tham Ming Quan with +6012-2799255.
