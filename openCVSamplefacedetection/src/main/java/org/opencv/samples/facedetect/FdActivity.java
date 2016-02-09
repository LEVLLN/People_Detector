package org.opencv.samples.facedetect;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class FdActivity extends Activity implements CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    public static final int JAVA_DETECTOR = 0;
    public static final int NATIVE_DETECTOR = 1;

    private MenuItem mItemFace50;
    private MenuItem mItemFace40;
    private MenuItem mItemFace30;
    private MenuItem mItemFace20;
    private MenuItem mSaveFace;
    private MenuItem mFacesToFiles;
    private MenuItem mItemType;
    private Rect finalFaceRect;
    private Mat mRgba;
    private Mat mGray;
    private File mFaceCascadeFile;
    private File mFullBodyCascadeFile;
    private CascadeClassifier mFaceDetector;
    private CascadeClassifier mFullBodyDetector;
    private DetectionBasedTracker mNativeDetector;
    private DetectionBasedTracker mNativeDetector_body;

    private int mDetectorType = JAVA_DETECTOR;
    private String[] mDetectorName;
    private List<Bitmap> bitmapList = new ArrayList<Bitmap>();
    private Bitmap bitmap;
    private float mRelativeFaceSize = 0.2f;
    private int mAbsoluteFaceSize = 0;

    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("detection_based_tracker");

                    try {
                        // load cascade file from application resources
                        InputStream isFace = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mFaceCascadeFile = new File(cascadeDir, "temp.xml");
                        FileOutputStream os = new FileOutputStream(mFaceCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = isFace.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        isFace.close();
                        os.close();

                        mFaceDetector = new CascadeClassifier(mFaceCascadeFile.getAbsolutePath());
                        if (mFaceDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mFaceDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mFaceCascadeFile.getAbsolutePath());

                        mNativeDetector = new DetectionBasedTracker(mFaceCascadeFile.getAbsolutePath(), 0);

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }
                    try {
                        // load cascade file from application resources
                        InputStream isFullBody = getResources().openRawResource(R.raw.haarcascade_fullbody);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mFullBodyCascadeFile = new File(cascadeDir, "temp.xml");
                        FileOutputStream os = new FileOutputStream(mFullBodyCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = isFullBody.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        isFullBody.close();
                        os.close();
                        mFullBodyDetector = new CascadeClassifier(mFullBodyCascadeFile.getAbsolutePath());
                        if (mFullBodyDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mFullBodyDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mFullBodyCascadeFile.getAbsolutePath());

                        mNativeDetector_body = new DetectionBasedTracker(mFullBodyCascadeFile.getAbsolutePath(), 0);

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }
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

    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
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

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
            mNativeDetector_body.setMinFaceSize(mAbsoluteFaceSize);
        }

        final MatOfRect faces = new MatOfRect();
        MatOfRect fullBodyes = new MatOfRect();

        if (mDetectorType == JAVA_DETECTOR) {
            if (mFaceDetector != null)
                mFaceDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
            if (mFullBodyDetector != null)
                mFullBodyDetector.detectMultiScale(mGray, fullBodyes, 1.1, 2, 2,
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        } else if (mDetectorType == NATIVE_DETECTOR) {
            if (mNativeDetector != null) {
                mNativeDetector.detect(mGray, faces);
                mNativeDetector_body.detect(mGray, fullBodyes);
            }
        } else {
            Log.e(TAG, "Detection method is not selected!");
        }

        final Rect[] facesArray = faces.toArray();
        Rect[] fullBodiesArray = fullBodyes.toArray();
        Rect faceRect = new Rect();
        for (final Rect aFacesArray : facesArray) {
            Imgproc.rectangle(mRgba, aFacesArray.tl(), aFacesArray.br(), FACE_RECT_COLOR, 3);
            faceRect = aFacesArray;


        }
        for (Rect aFullBodiesArray : fullBodiesArray) {
            Imgproc.rectangle(mRgba, aFullBodiesArray.tl(), aFullBodiesArray.br(), FACE_RECT_COLOR, 3);
        }


        finalFaceRect = faceRect;
        mOpenCvCameraView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {


            }
        });
        return mRgba;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        mSaveFace = menu.add("Remember Face");
        mFacesToFiles = menu.add("Faces to Storage");
        mItemType = menu.add(mDetectorName[mDetectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);
        else if (item == mItemType) {
            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[tmpDetectorType]);
            setDetectorType(tmpDetectorType);
        } else if (item == mSaveFace) {
            Mat faceRecMat = mRgba.submat(finalFaceRect.y, finalFaceRect.y + finalFaceRect.height, finalFaceRect.x, finalFaceRect.x + finalFaceRect.width);
            bitmap = Bitmap.createBitmap(faceRecMat.cols(), faceRecMat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(faceRecMat, bitmap);
            bitmapList.add(bitmap);
        } else if (item == mFacesToFiles) {
            try {
                String path = Environment.getExternalStorageDirectory().toString();

                for (Bitmap k : bitmapList) {
                    OutputStream fOut = null;
                    Random rnd = new Random();
                    int number = rnd.nextInt(100 + 300);
                    File file = new File(path, "Finded face" + number + ".png"); // the File to save to jpg
                    fOut = new FileOutputStream(file);
                    k.compress(Bitmap.CompressFormat.JPEG, 100, fOut);
                    fOut.flush();
                    fOut.close(); // do not forget to close the stream
                    MediaStore.Images.Media.insertImage(getContentResolver(), file.getAbsolutePath(), file.getName(), file.getName());
                    Log.i("SAVED", "FACE " + number + " SAVED");
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                Log.i(TAG, "Detection Based Tracker enabled");
                mNativeDetector.start();
            } else {
                Log.i(TAG, "Cascade detector enabled");
                mNativeDetector.stop();
            }
        }
    }
}
