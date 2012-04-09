#define CHECK_RC(rc, what)											\
	if (rc != XN_STATUS_OK)											\
	{																\
		printf("%s failed: %s\n", what, xnGetStatusString(rc));		\
		return rc;													\
	}


//---------------------------------------------------------------------------
// Includes
//---------------------------------------------------------------------------
#include <XnOpenNI.h>
#include <XnCppWrapper.h>
#include <XnPropNames.h>
#include <cv.h>
#include <highgui.h>

#define SAMPLE_XML_PATH "./SamplesConfig.xml"
#define MAX_DEPTH 10000

using namespace xn;
using namespace cv;

bool isDebugConf = false;

// AWESOME TEMPLATE TO GET DIRECT ACCESS TO IMAGES
template<class T> class Image {
    private:
        IplImage* imgp;
    public:
        Image(IplImage* img=0) {imgp=img;}
        ~Image(){imgp=0;}

    void operator=(IplImage* img) {imgp=img;}
    inline T* operator[](const int rowIndx) {
    return ((T *)(imgp->imageData + rowIndx*imgp->widthStep));}
};

typedef struct{ unsigned char b,g,r; short x,y;} RgbPixel;
typedef struct{ float b,g,r; } RgbPixelFloat;
typedef Image<RgbPixel> RgbImage;
typedef Image<RgbPixelFloat> RgbImageFloat;
typedef Image<unsigned char>BwImage;
typedef Image<float> BwImageFloat;
typedef Image<unsigned short int>Bw2Image;
typedef struct{ double x,y,z;} P3D;
typedef struct{ double x,y;} P2D;

Bw2Image depthImage;
RgbImage rgbImage;

Context context;
DepthGenerator depth;
ImageGenerator image;
DepthMetaData depthMD;
ImageMetaData imageMD;
XnStatus rc;
Player player;

// tmp do get data from kinect to openCV
CvMat* depthMetersMat;
IplImage *kinectDepthImage;
CvMat* depthMetersMat2;
IplImage *kinectDepthImage2;

cv::Mat colorArr[3];
cv::Mat colorImage;
const XnRGB24Pixel* pImageRow;
const XnRGB24Pixel* pPixel;
const XnDepthPixel* pDepth;

IplImage iplImage;

// recording stuff
const char* strInputFile = "./bin/Data/recording1.oni";
const char* strOutputFile = "./bin/Data/output1.avi";
const char* strPathToXML = "/home/pawe/kinect/project/bin/Debug/SamplesConfig.xml";

CvVideoWriter  * avi = NULL;
XnUInt32 nNumFrames = 0;

bool drawLines = false;
const int nbOfPoints = 20;
int const nbOfMarkers = 9;
int const proximityLimit = 16;

RgbPixel markers[nbOfMarkers];
CvPoint defaults[nbOfMarkers];
bool initialize = true;

short maxNeighbSize = 30;

IplImage* imgThreshed;
IplImage* faceImage;

void paintMarkerOnBoth(RgbPixel rgbPixel);
void paintFace();
CvPoint intoDepthCVPoint(CvPoint);
RgbPixel getCurrentMarker( int );
void setCurrentMarker(RgbPixel, int);
int intoDepthX(int);
int intoDepthY(int);

class MyPoint {
public:
  float x,y;
  MyPoint(float xx, float yy) { x = xx; y=yy; }
  float getX() { return x; }
  float getY() { return y; }

};

MyPoint points[nbOfPoints] {
//  point 0 TOP LEFT
(MyPoint(0,0)),
//  point 1 BRODA
(MyPoint(55,135)),
//  point 2 TOP RIGHT
(MyPoint(110,0)),
//  point 3 BOTTOM LEFT
(MyPoint(0,90)),
//  point 4 BOTTOM RIGHT
(MyPoint(110,90)),
//  point 5 EYEB_L_I
(MyPoint(40,10)),
//  point 6 EYEB_R_I
(MyPoint(80,10)),
//  point 7 EYEB_L_O
(MyPoint(15,20)),
//  point 8 EYEB_R_O
(MyPoint(55,135)),
//  point 9 EYE_L_UP
(MyPoint(30,30)),
//  point 10 EYE_R_UP
(MyPoint(90,30)),
//  point 11 EYE_L_DOWN
(MyPoint(30,40)),
//  point 12 EYE_R_DOWN
(MyPoint(90,40)),
//  point 13 NOSE
(MyPoint(55,70)),
//  point 14 CHIN_L
(MyPoint(20,80)),
//  point 15 CHIN_R
(MyPoint(90,80)),
//  point 16 MOUTH_UP
(MyPoint(55,90)),
//  point 17 MOUTH L
(MyPoint(30,95)),
//  point 18 MOUTH_R
(MyPoint(80,95)),
//  point 19 CHIN
(MyPoint(55,135)),
};


void normalizeReferenceFace() {

    short shift_up = 0;
    float heigth = abs(points[1].y - points[0].y);
    float width = (points[1].x-points[0].x)*2;

    for ( int i = 0; i < nbOfPoints; i++) {
        points[i].x = points[i].x/width;
        points[i].y = points[i].y/heigth + ((float)shift_up)/heigth;
        //printf("POINTS %f\n",points[i].x);
    }
}

int main(int argc, char* argv[])
{

	EnumerationErrors errors;


    //rc = context.Init();
    rc = context.InitFromXmlFile(strPathToXML,&errors);
    if (rc == XN_STATUS_NO_NODE_PRESENT)
	{
		XnChar strError[1024];
		errors.ToString(strError, 1024);
		printf("%s\n", strError);
		return (rc);
	}
	else if (rc != XN_STATUS_OK)
	{
		printf("Open failed: %s\n", xnGetStatusString(rc));
		return (rc);
	}
	
	/* UNCOMMENT TO GET FILE READING 
    //rc = context.OpenFileRecording(strInputFile);
	//CHECK_RC(rc, "Open input file");

	//rc = context.FindExistingNode(XN_NODE_TYPE_PLAYER, player);
	//CHECK_RC(rc, "Get player node"); */ 

	rc = context.FindExistingNode(XN_NODE_TYPE_DEPTH, depth);
	CHECK_RC(rc, "Find depth generator");

	rc = context.FindExistingNode(XN_NODE_TYPE_IMAGE, image);
	CHECK_RC(rc, "Find image generator");

    depth.GetMetaData(depthMD);
	image.GetMetaData(imageMD);

    //rc = player.SetRepeat(FALSE);
	XN_IS_STATUS_OK(rc);

    //rc = player.GetNumFrames(image.GetName(), nNumFrames);
	//CHECK_RC(rc, "Get player number of frames");
	//printf("%d\n",nNumFrames);

    //rc = player.GetNumFrames(depth.GetName(), nNumFrames);
	//CHECK_RC(rc, "Get player number of frames");
	//printf("%d\n",nNumFrames);

	// Hybrid mode isn't supported
	if (imageMD.FullXRes() != depthMD.FullXRes() || imageMD.FullYRes() != depthMD.FullYRes())
	{
		printf ("The device depth and image resolution must be equal!\n");
		return 1;
	}

	// RGB is the only image format supported.
	if (imageMD.PixelFormat() != XN_PIXEL_FORMAT_RGB24)
	{
		printf("The device image format must be RGB24\n");
		return 1;
	}

    avi = cvCreateVideoWriter(strOutputFile, 0, 30, cvSize(640,480), TRUE);

    depthMetersMat = cvCreateMat(480, 640, CV_16UC1);
    kinectDepthImage = cvCreateImage( cvSize(640,480),16,1 );

    depthMetersMat2 = cvCreateMat(480, 640, CV_16UC1);
    kinectDepthImage2 = cvCreateImage( cvSize(640,480),16,1 );

    colorArr[0] = cv::Mat(imageMD.YRes(),imageMD.XRes(),CV_8U);
    colorArr[1] = cv::Mat(imageMD.YRes(),imageMD.XRes(),CV_8U);
    colorArr[2] = cv::Mat(imageMD.YRes(),imageMD.XRes(),CV_8U);

    //prepare_for_face_detection();

    int b;
    int g;
    int r;

	while ((rc = image.WaitAndUpdateData()) != XN_STATUS_EOF && (rc = depth.WaitAndUpdateData()) != XN_STATUS_EOF) {
        if (rc != XN_STATUS_OK) {
            printf("Read failed: %s\n", xnGetStatusString(rc));
            break;
        }
        depth.GetMetaData(depthMD);
        image.GetMetaData(imageMD);

        //XnUInt32 a;
        //a = g_imageMD.FPS;
        printf("%d\n",imageMD.FrameID());
        //a = g_depthMD.DataSize();
        //printf("%d\n",a);

        pDepth = depthMD.Data();
        pImageRow = imageMD.RGB24Data();

        for (unsigned int y=0; y<imageMD.YRes(); y++) {
            pPixel = pImageRow;
            uchar* Bptr = colorArr[0].ptr<uchar>(y);
            uchar* Gptr = colorArr[1].ptr<uchar>(y);
            uchar* Rptr = colorArr[2].ptr<uchar>(y);

            for(unsigned int x=0;x<imageMD.XRes();++x , ++pPixel){
                Bptr[x] = pPixel->nBlue;
                Gptr[x] = pPixel->nGreen;
                Rptr[x] = pPixel->nRed;

                depthMetersMat->data.s[y * XN_VGA_X_RES + x ] = 7*pDepth[y * XN_VGA_X_RES + x];
                depthMetersMat2->data.s[y * XN_VGA_X_RES + x ] = pDepth[y * XN_VGA_X_RES + x];
            }
            pImageRow += imageMD.XRes();
        }
        cv::merge(colorArr,3,colorImage);
        iplImage = colorImage;

        //cvThreshold(depthMetersMat2, depthMetersMat2, 150, 1500, THRESH_BINARY);

        cvGetImage(depthMetersMat,kinectDepthImage);
        cvGetImage(depthMetersMat2,kinectDepthImage2);

        depthImage = Bw2Image(kinectDepthImage2);
        printf("1. Middle pixel is %u millimeters away\n",depthImage[240][320]);

        rgbImage = RgbImage(&iplImage);

		// we want to see on up to 2000 MM 
        int THRESH = 2000;

        for (unsigned int y=0; y<imageMD.YRes(); y++) {
            for(unsigned int x=0;x<imageMD.XRes();++x){
                if ( depthImage[y][x] >= THRESH ) {
                    depthImage[y][x] = 0;
                } else {
                    float tmp = depthImage[y][x];
                    tmp = tmp / THRESH * (65536)*(-1) + 65536;
                    depthImage[y][x] = (unsigned int)tmp;
                }
            }
        }
		
		// THE PART ABOUT FILTERING COLOURS IN HSV TO SEE ONLY SPECIFIC ONE 
		// AFTER ONE FEW MORPHOLOGICAL OPERATIONS TO MAKE IT LOOK BETTER 

        IplImage* imgHSV = cvCreateImage(cvGetSize(&iplImage), 8, 3);
        cvCvtColor(&iplImage, imgHSV, CV_BGR2HSV);
        imgThreshed = cvCreateImage(cvGetSize(&iplImage), 8, 1);
        //cvInRangeS(imgHSV, cvScalar(100, 60, 80), cvScalar(110, 255, 255), imgThreshed); // BLUE
        cvInRangeS(imgHSV, cvScalar(29, 95, 95), cvScalar(35, 255, 255), imgThreshed); // YELLOW
        //cvInRangeS(imgHSV, cvScalar(29, 60, 60), cvScalar(35, 255, 255), imgThreshed); // YELLOW DARK
        //cvInRangeS(imgHSV, cvScalar(150, 70, 70), cvScalar(160, 255, 255), imgThreshed); // PINK
        //cvInRangeS(imgHSV, cvScalar(40, 76, 76), cvScalar(70, 255, 255), imgThreshed); // GREEN
        IplConvKernel* kernel = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_RECT, NULL);
        //cvDilate(imgThreshed,imgThreshed,kernel);
        //cvErode(imgThreshed,imgThreshed,kernel);
        Mat mat = Mat(imgThreshed);
        blur(Mat(imgThreshed),mat,cvSize(3,3));
        imgThreshed = &IplImage(mat);
        //cvInRangeS(imgThreshed,cvScalar(100),cvScalar(255),imgThreshed);
        //cvErode(imgThreshed,imgThreshed,kernel);
        cvDilate(imgThreshed,imgThreshed,kernel);
        cvDilate(imgThreshed,imgThreshed,kernel);
        cvErode(imgThreshed,imgThreshed,kernel);
        cvErode(imgThreshed,imgThreshed,kernel);
        mat = Mat(imgThreshed);
        blur(Mat(imgThreshed),mat,cvSize(6,6));
        imgThreshed = &IplImage(mat);
        cvInRangeS(imgThreshed,cvScalar(100),cvScalar(255),imgThreshed);
        cvReleaseImage(&imgHSV);
        BwImage threshed = BwImage(imgThreshed);



        if ( initialize == true ) {

            normalizeReferenceFace();
            int currentID = 0;

                for ( int y = 30; y<480; y++ ) {
                    for ( int x = 30; x<640; x++ ) {
                        bool g2g = true;
                        //printf("%d %d %d\n",ID, y,x);
                        if ( threshed[y][x]!=0 ) {
                            for ( int ID2 = 0; ID2<nbOfPoints; ID2++) {
                                if ( (abs(markers[ID2].y-y)<proximityLimit) && (abs(markers[ID2].x-x)<proximityLimit)) {
                                    g2g = false;
                                }
                            }
                            if (currentID >= nbOfPoints || g2g == false ) {
                                break;
                            }
                            markers[currentID].y=y;
                            markers[currentID].x=x;
                            currentID++;
                            printf("WHITE PIXEL INITIALIZED %d: %d %d\n",currentID, x,y);
                        }
                    }
                }


            if (isDebugConf==true || currentID == nbOfMarkers) {
                printf("%d PIXELS INITIALIZED\n", currentID);
                initialize = false;
                //printf("%d,%d\n", currentID, nbOfPoints);
                //return 0;
            } else {
                printf("WAITING FOR %d PIXELS TO APPEAR, %d SO FAR \n",nbOfMarkers, currentID);
                continue;
            }


            // FIND TOP RIGHT AND CHIN PIXEL

            int refPixID = 0;
            int chinPixID = 0;

            for ( int i = 0; i < nbOfMarkers; i++) {
                if ( (markers[i].x + markers[i].y)*(markers[i].x + markers[i].y) < (markers[refPixID].x + markers[refPixID].y)* (markers[refPixID].x + markers[refPixID].y)) {
                    refPixID = i;
                }
                if (markers[i].y > markers[chinPixID].y) {
                    chinPixID = i;
                }
            }

            float width = (markers[1].x-markers[0].x)*2;
            float heigth = abs(markers[1].y-markers[0].y);

            // WE GOT WIDTH & HEIGTH OF THE FACE, LETS ADJUST POINTS

            // SET 0 to REF, SET 1 to CHIN

            MyPoint tmp = MyPoint(markers[refPixID].x,markers[refPixID].y);
            markers[refPixID].x = markers[0].x;
            markers[refPixID].y = markers[0].y;
            markers[0].x = tmp.x;
            markers[0].y = tmp.y;

            tmp = MyPoint(markers[chinPixID].x,markers[chinPixID].y);
            markers[chinPixID].x = markers[1].x;
            markers[chinPixID].y = markers[1].y;
            markers[1].x = tmp.x;
            markers[1].y = tmp.y;


            // REST OF THE POINTS

            for ( int i = 2; i < nbOfPoints; i++) {

                int cost = 0;
                int lowestCost = 0;
                int closestPixID = -1;


                for ( int j = 2; j < nbOfMarkers; j++ ) {
                    cost = (markers[j].x-points[i].x*width)*(markers[j].x-points[i].x*width) + (markers[j].y-points[i].y*heigth)*(markers[j].y-points[i].y*heigth);
                    if ( cost < lowestCost ) {
                        lowestCost = cost;
                        closestPixID = j;
                    }
                    if (closestPixID == -1) {
                        //printf("COS JEST SPORO NIE W PORZADKU, CHECK HERE\n");
                        break;
                    }
                    tmp.x = markers[i].x;
                    tmp.y = markers[i].y;
                    markers[i].x=markers[closestPixID].x;
                    markers[i].x=markers[closestPixID].y;
                    markers[closestPixID].x = tmp.x;
                    markers[closestPixID].y = tmp.y;
                }
            }
        }

        for ( int currentPixelID = 0; currentPixelID < nbOfMarkers; currentPixelID++) {
            if (markers[currentPixelID].x == 0) {
                continue;
            }

            if ( threshed[markers[currentPixelID].y][markers[currentPixelID].x] < 128 ) {
                printf("PIXEL %d LOST\n",currentPixelID);

                for ( int neighbSize = 2; neighbSize < maxNeighbSize; neighbSize = neighbSize + 2 ) {

                    int x1 = markers[currentPixelID].x - neighbSize/2;
                    if ( x1 < intoDepthX(0) ) {
                        x1 = (int)intoDepthX(0);
                    }

                    int y1 = (int)(markers[currentPixelID].y-neighbSize/2);
                    if (  y1 < intoDepthY(0) ) {
                        y1 = intoDepthY(0);
                    }

                    int y2 = markers[currentPixelID].y+neighbSize/2;
                    if (  y2 > intoDepthY(480)  ) {
                        y2 = intoDepthY(480);
                    }

                    int x2 = markers[currentPixelID].x+neighbSize/2;
                    if ( x2 > intoDepthX(640) ) {
                        y2 = intoDepthX(640);
                    }

                    bool found = false;
                    for ( int y = y1; y < y2; y++) {
                        for ( int x = x1; x < x2; x++) {
                            bool g2g = true;
                            if (threshed[y][x] > 128) {
                                for ( int ID2 = 0; ID2<nbOfMarkers; ID2++) {
                                    if ( currentPixelID == ID2 )
                                        continue;
                                    if ( (abs(markers[ID2].y-y)<proximityLimit) && (abs(markers[ID2].x-x)<proximityLimit)) {
                                        g2g = false;
                                        break;
                                    }
                                }

                                if ( g2g ) {
                                    markers[currentPixelID].x = x;
                                    markers[currentPixelID].y = y;
                                    found = true;
                                    printf("Pixel %d, FOUND\n",currentPixelID);
                                    break;
                                }
                            }
                        }
                        if (found == true ) {
                            break;
                        }
                    }
                    if (found == true ) {
                        break;
                    }
                }
            }

            paintMarkerOnBoth(markers[currentPixelID]);

        }
        faceImage = cvCreateImage(cvGetSize(&iplImage), 8, 1);
        paintFace();

		// normal kinect depth
        cvShowImage("Depth_Kinect", kinectDepthImage);
		// depth within 80 - 200 mm, normalized 
        cvShowImage("Depth_Kinect_2", kinectDepthImage2);
		// rgb with tracking points
        cvShowImage("RGB_Kinect", &iplImage);
		// colour detector 
        cvShowImage("RGB_Threshed", imgThreshed);
		// attempt to draw a face 
        cvShowImage("Face Image", faceImage);

        cvWaitKey(50);           // wait 20 ms

        if ( avi == NULL) {
            printf ("dupa%d \n",1);
        }
        //cvWriteFrame (avi, &iplImage);
	}

//    cvReleaseImageHeader(kinectDepthImage);
    cvReleaseVideoWriter(&avi);
//    cvReleaseHaarClassifierCascade( &cascade );
    context.Shutdown();

	return 0;
}

void setCurrentMarker(RgbPixel rgbPixel, int currentPixelID) {
    markers[currentPixelID].x = rgbPixel.x;
    markers[currentPixelID].y = rgbPixel.y;
}


RgbPixel getCurrentMarker(int currentPixelID) {

    RgbPixel rgbPixel;

    rgbPixel.x = markers[currentPixelID].x;
    rgbPixel.y = markers[currentPixelID].y;

    return rgbPixel;

}

void paintFace() {

    for (int i = 0; i<nbOfMarkers; i++) {
        CvPoint point = cvPoint(markers[i].x,markers[i].y);
        cvCircle(faceImage,point, 2, cvScalar(65000,65000,65000), 3, 8);
    }

    //printf("%d,%d,%d\n",rgbImage[markers[0].x][markers[0].y].b,rgbImage[markers[0].x][markers[0].y].g,rgbImage[markers[0].x][markers[0].y].r);

    //eyes
    //cvCircle(faceImage,cvPoint(markers[9].x,markers[9].y-markers[11].y), 8, cvScalar(65000,65000,65000), 3, 8);
    //cvCircle(faceImage,cvPoint(markers[10].x,markers[10].y-markers[12].y), 8, cvScalar(65000,65000,65000), 3, 8);

    // nose
    //cvCircle(faceImage,cvPoint(markers[5].x,markers[5].y), 8, cvScalar(65000,65000,65000), 8, 8);

    // mouth
    //cvEllipse(faceImage,cvPoint(markers[16].x,markers[16].y-markers[19].y),cvSize(abs(markers[18].x-markers[17].x),abs(markers[19].y-markers[16].y)),0,0,360,cvScalar(65000,65000,65000),5,5 );


    // contours and eyebrows
    if ( drawLines==true) {
        int p1 = 0;
        int p2 = 3;
        cvLine(faceImage, cvPoint(markers[p1].x,markers[p1].y),cvPoint(markers[p2].x,markers[p2].y),cvScalar(65000,65000,65000));
        p1 = 4;
        p2 = 2;
        cvLine(faceImage, cvPoint(markers[p1].x,markers[p1].y),cvPoint(markers[p2].x,markers[p2].y),cvScalar(65000,65000,65000));
        p1 = 3;
        p2 = 1;
        cvLine(faceImage, cvPoint(markers[p1].x,markers[p1].y),cvPoint(markers[p2].x,markers[p2].y),cvScalar(65000,65000,65000));
        p1 = 1;
        p2 = 2;
        cvLine(faceImage, cvPoint(markers[p1].x,markers[p1].y),cvPoint(markers[p2].x,markers[p2].y),cvScalar(65000,65000,65000));
    }
    /*
    int p1 = 0;
    int p2 = 2;
    cvLine(faceImage, cvPoint(markers[p1].x,markers[p1].y),cvPoint(markers[p2].x,markers[p2].y),cvScalar(65000,65000,65000));
    p1 = 0;
    p2 = 3;
    cvLine(faceImage, cvPoint(markers[p1].x,markers[p1].y),cvPoint(markers[p2].x,markers[p2].y),cvScalar(65000,65000,65000));
    p1 = 3;
    p2 = 1;
    cvLine(faceImage, cvPoint(markers[p1].x,markers[p1].y),cvPoint(markers[p2].x,markers[p2].y),cvScalar(65000,65000,65000));
    p1 = 1;
    p2 = 4;
    cvLine(faceImage, cvPoint(markers[p1].x,markers[p1].y),cvPoint(markers[p2].x,markers[p2].y),cvScalar(65000,65000,65000)); */
    //p1 = 4;
    //p2 = 2;
    //cvLine(faceImage, cvPoint(markers[p1].x,markers[p1].y),cvPoint(markers[p2].x,markers[p2].y),cvScalar(65000,65000,65000));
    //p1 = 7;
    //p2 = 5;
    //cvLine(faceImage, cvPoint(markers[p1].x,markers[p1].y),cvPoint(markers[p2].x,markers[p2].y),cvScalar(65000,65000,65000));
    //p1 = 6;
    //p2 = 8;
    //cvLine(faceImage, cvPoint(markers[p1].x,markers[p1].y),cvPoint(markers[p2].x,markers[p2].y),cvScalar(65000,65000,65000));

}

void paintMarkerOnBoth(RgbPixel rgbPixel) {

    CvPoint point;
    point = cvPoint(rgbPixel.x, rgbPixel.y);

    int size = 1;
    if ( depthImage[rgbPixel.x][rgbPixel.y] < 800 ) {
        size = 10;
    } else if ( depthImage[rgbPixel.x][rgbPixel.y] < 900 ) {
        size = 9;
    } else if ( depthImage[rgbPixel.x][rgbPixel.y] < 1000 ) {
        size = 8;
    } else if ( depthImage[rgbPixel.x][rgbPixel.y] < 1100 ) {
        size = 7;
    } else if ( depthImage[rgbPixel.x][rgbPixel.y] < 1200 ) {
        size = 6;
    } else if ( depthImage[rgbPixel.x][rgbPixel.y] < 1300 ) {
        size = 5;
    } else if ( depthImage[rgbPixel.x][rgbPixel.y] < 1400 ) {
        size = 4;
    } else if ( depthImage[rgbPixel.x][rgbPixel.y] < 1500 ) {
        size = 3;
    } else if ( depthImage[rgbPixel.x][rgbPixel.y] < 1600 ) {
        size = 2;
    } else if ( depthImage[rgbPixel.x][rgbPixel.y] < 1700 ) {
        size = 1;
    };

    //printf("DISTANCE %d X = %d\n ",depthImage[rgbPixel.x][rgbPixel.y],rgbPixel.x);


    cvCircle(&iplImage,point, 2, cvScalar(0,0,255), 3, 8);
    cvCircle(kinectDepthImage, intoDepthCVPoint(point), 2, cvScalar(65000,65000,65000), -2, 8);
    cvCircle(kinectDepthImage2, intoDepthCVPoint(point), 2, cvScalar(2,0,0), 3, 8);

    //cvRectangle(&iplImage, cvPoint(rgbPixel.x-2, rgbPixel.y-2), cvPoint(rgbPixel.x+2, rgbPixel.y+2),cvScalar(255,0,0), 2);
    //cvRectangle(kinectDepthImage, cvPoint(rgbPixel.x-2, rgbPixel.y-2), cvPoint(rgbPixel.x+2, rgbPixel.y+2),cvScalar(65000,65000,65000), 2);
}

int intoDepthX(int x) {
    return (double)abs(x - 46)/586*640;
}

int intoDepthY(int y) {
    return (double)abs(y - 37)/436*480;
}

CvPoint intoDepthCVPoint(CvPoint point) {
    return cvPoint(intoDepthX(point.x),intoDepthY(point.y));
}

MyPoint intoDepthMyPoint(MyPoint point) {
    return MyPoint(intoDepthX(point.x),intoDepthY(point.y));
}

            // FIELD OF VIEW OF DEPTH
            //int x1 = 37;
            //int y1 = 30;
            //int x2 = 632;
            //int y2 = 480;

/* MY OWN CALIBRATION
            int x1 = 125; // rec1 = 225 2 = 152 3 = 165
            int y1 = 62; // rec1 = 132 2 = 87 3 = 82
            int x2 = 525; // rec1 = 445 2 = 475 3 = 505
            int y2 = 413; // rec1 = 347 2 = 387 3 = 393
            int dx1 = (double)(x1 - 46)/586*640; // 2 = 45/587
            int dy1 = (double)(y1 - 37)/436*480; // 2 = 37/436
            int dx2 = (double)(x2 - 46)/586*640;
            int dy2 = (double)(y2 - 37)/436*480;
            //int dx1 = x1 - 23;
            //int dy1 = y1 - 30;
            //int dx2 = x2;
            //int dy2 = y2 - 6;
            cvRectangle(&iplImage, cvPoint(x1, y1), cvPoint(x2,y2),cvScalar(65000,65000,65000), 1);
            cvRectangle(kinectDepthImage2, cvPoint(dx1, dy1), cvPoint(dx2, dy2),cvScalar(65000,65000,65000), 1);

            */


/* OLD TRACKING
        for ( int currentPixelID = 0; currentPixelID < nbOfMarkers; currentPixelID++) {
            if (markers[currentPixelID].x == 0) {
                continue;
            }


            RgbPixel rgbPixel = getCurrentMarker(currentPixelID);

            // check the value of reference pixel and next pixel at same position
            int selfCost = abs(rgbPixel.b - rgbImage[rgbPixel.y][rgbPixel.x].b) + abs(rgbPixel.g - rgbImage[rgbPixel.y][rgbPixel.x].g) + abs(rgbPixel.r - rgbImage[rgbPixel.y][rgbPixel.x].r);
            printf("Pixel %d, SELFCOST = %d \n", currentPixelID, selfCost);

            bool adjustToBrightness = false;

            short adjThresh = 5;
            if ( ( abs(rgbPixel.b - rgbImage[rgbPixel.y][rgbPixel.x].b) < adjThresh ) && ( abs(rgbPixel.g - rgbImage[rgbPixel.y][rgbPixel.x].g) < adjThresh ) && ( abs(rgbPixel.r - rgbImage[rgbPixel.y][rgbPixel.x].r) < adjThresh )) {
                adjustToBrightness = true;
                printf("Pixel %d: Adjusting to brightness change. ",currentPixelID);
            }

            if ( (selfCost > 4) && adjustToBrightness == false) {
                bool doAgain = false;
                do {
                    RgbPixel tmp;
                    tmp.x = rgbPixel.x;
                    tmp.y = rgbPixel.y;
                    tmp.b = rgbPixel.b;
                    tmp.g = rgbPixel.g;
                    tmp.r = rgbPixel.r;

                    int currentCost = selfCost;
                    int cost = 756;

                    int x1 = rgbPixel.x - neighbSize/2;
                    if ( x1 < intoDepthX(0) ) {
                        x1 = (int)intoDepthX(0);
                    }
                    int y1 = rgbPixel.y-neighbSize/2;
                    if (  y1 < intoDepthY(0) ) {
                        y1 = intoDepthY(0);
                    }
                    int y2 = rgbPixel.y+neighbSize/2;
                    if (  y2 > intoDepthY(480)  ) {
                        y2 = intoDepthY(480);
                    }
                    int x2 = rgbPixel.x+neighbSize/2;
                    if ( x2 > intoDepthX(640) ) {
                        y2 = intoDepthX(640);
                    }

                    //for ( short y = rgbPixel.y-neighbSize/2+rgbPixel.dy/(neighbSize/2); y < ( rgbPixel.y+neighbSize/2+rgbPixel.dy/(neighbSize/2) ); y++) {
                    //    for ( short x = rgbPixel.x-neighbSize/2 + rgbPixel.dx/(neighbSize/2); x < ( rgbPixel.x+neighbSize/2 + rgbPixel.dx/(neighbSize/2)); x++) {
                    for ( int y = y1; y < y2; y++) {
                        for ( int x = x1; x < x2; x++) {
                            cost = abs(tmp.b - rgbImage[y][x].b) + abs(tmp.g - rgbImage[y][x].g) + abs(tmp.r - rgbImage[y][x].r);
                            //printf("Pixel %d: COST = %d\n",currentPixelID,cost);
                            if (cost < currentCost) {
                                currentCost = cost;
                                tmp.x = x;
                                tmp.y = y;
                                tmp.b = rgbImage[x][y].b;
                                tmp.g = rgbImage[x][y].g;
                                tmp.r = rgbImage[x][y].r;
                            }
                        }
                    }
                    int thresholdCost = 200;
                    if ( currentCost < thresholdCost && (abs(selfCost-currentCost)>1)) {
                        //printf("%d\n",selfCost);
                        if ( doAgain == false ) {
                            rgbPixel.dx = tmp.x-rgbPixel.x;
                            rgbPixel.dy = tmp.y-rgbPixel.y;
                         }
                        rgbPixel.x = tmp.x;
                        rgbPixel.y = tmp.y;
                        rgbPixel.b = tmp.b;
                        rgbPixel.g = tmp.g;
                        rgbPixel.r = tmp.r;
                        doAgain=false;
                    } /*else {
                        neighbSize=neighbSize+dNeighbSize;
                        doAgain = true;
                        printf("Pixel %d, INCREASING NEIGHBOURHOOD TO %d\n",currentPixelID,neighbSize);
                    }
                    if (((int)tmp.x-neighbSize/2 < 0) || ((int)tmp.y-neighbSize/2 < 0) || ((int)tmp.x+neighbSize/2>(int)imageMD.XRes()) || ((int)tmp.x+neighbSize/2>(int)imageMD.XRes()) ) {
                        doAgain = false;
                        //initialize = true;
                        printf("Pixel %d, PIXEL LOST, SKIPPING\n",currentPixelID);
                    }
                }
                while ( doAgain && (neighbSize < 60) );
            } else {
                printf("Pixel %d, No NEED to CHANGE, adjusting values\n", currentPixelID);
                rgbPixel.b = rgbImage[rgbPixel.y][rgbPixel.x].b;
                rgbPixel.g = rgbImage[rgbPixel.y][rgbPixel.x].g;
                rgbPixel.r = rgbImage[rgbPixel.y][rgbPixel.x].r;
            }

            setCurrentMarker(rgbPixel, currentPixelID);

            paintMarkerOnBoth(markers[currentPixelID]);


        }

        */
/* NICOAS BURRUS STUFF !
            double fx_rgb =5.2921508098293293e+02;
            double fy_rgb =5.2556393630057437e+02;
            double cx_rgb =3.2894272028759258e+02;
            double cy_rgb =2.6748068171871557e+02;

            double fx_d =5.9421434211923247e+02;
            double fy_d =5.9104053696870778e+02;
            double cx_d =3.3930780975300314e+02;
            double cy_d =2.4273913761751615e+02;

            float x_d = 200;
            float y_d = 300;

            P3D p3d;
            p3d.x = ((double)x_d - cx_d) * (double)((double)((int)(depthImage[x_d,y_d])/1000)) / fx_d;
            p3d.y = ((double)y_d - cy_d) * (double)((int)depthImage[x_d,y_d]/1000) / fy_d;
            p3d.z = (double)((int)depthImage[x_d,y_d]/1000 );

            P3D p3dn;
            P2D p2d_rgb;
//            p3dn = p3d+ T;
            p3dn.x = p3d.x * 1.9985242312092553e-02;
            p3dn.y = p3d.y * -7.4423738761617583e-04;
            p3dn.z = p3d.z * -1.0916736334336222e-02;
            p2d_rgb.x = (p3dn.x * fx_rgb / p3dn.z) + cx_rgb;
            p2d_rgb.y = (p3dn.y * fy_rgb / p3dn.z) + cy_rgb;

            //int XX = (p2d_rgb.x - cx_rgb)* p3dn.z / fx_rgb;
            //int YY = (p2d_rgb.x - cx_rgb)* p3dn.z / fx_rgb;

            float XX = (p2d_rgb.x - cx_rgb)* p3dn.z / fx_rgb;
            XX = XX/ 1.9985242312092553e-02;
            XX = XX*fx_d;
            XX = XX / (double)((unsigned int)depthImage[x_d,y_d])/1000;
            XX = XX - cx_d;
            float YY = (p2d_rgb.y - cy_rgb)* p3dn.z / fy_rgb;
            YY = YY/ -7.4423738761617583e-04;

            printf( "P3D.X %f, P3D.Y %f, P3D.Z %f, P2D.X %f, P2D.X %f \n", p3d.x, p3d.y,p3d.z,p2d_rgb.x,p2d_rgb.y);
            printf( " XX = %f YY = %f\n", XX, YY);


            */


/*
    void prepare_for_face_detection () {

        // Create memory for calculations
        storage = 0;

        // Create a new Haar classifier
        cascade = 0;

        // Load the HaarClassifierCascade
        cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );

        // Check whether the cascade has loaded successfully. Else report and error and quit
        if( !cascade ) {
            fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
            return;
        }

        // Allocate the memory storage
        storage = cvCreateMemStorage(0);

        // Create a new named window with title: result
        cvNamedWindow( "result", 1 );

        // Clear the memory storage which was used before
        cvClearMemStorage( storage );

        // Find whether the cascade is loaded, to find the faces. If yes, then:
    }



	void detect_and_draw( IplImage* img, IplImage* img2 ) {


    if( cascade )
    {

        IplImage* small_image = img;
        small_image = cvCreateImage( cvSize(img->width/2,img->height/2), IPL_DEPTH_8U, 3 );
        cvPyrDown( img, small_image, CV_GAUSSIAN_5x5 );
        int scale = 2;
        // There can be more than one face in an image. So create a growable sequence of faces.
        // Detect the objects and store them in the sequence
        CvSeq* faces = cvHaarDetectObjects( small_image, cascade, storage, 1.3, 2, CV_HAAR_DO_CANNY_PRUNING , cvSize(24, 24) );

        // Loop the number of faces found.
        for( int i = 0; i < (faces ? faces->total : 0); i++ )
        {
           // Create a new rectangle for drawing the face
            CvRect* r = (CvRect*)cvGetSeqElem( faces, i );

            // Find the dimensions of the face,and scale it if necessary
            pt1.x = r->x*scale;
            pt2.x = (r->x+r->width)*scale;
            pt1.y = r->y*scale;
            pt2.y = (r->y+r->height)*scale;

            // Draw the rectangle in the input image
            cvRectangle( img2, pt1, pt2, CV_RGB(255,0,0), 3, 8, 0 );
            cvRectangle( img, pt1, pt2, CV_RGB(255,0,0), 3, 8, 0 );
        }
    }

}
*/

//---------------------------------------------------------------------------
// Code
//---------------------------------------------------------------------------
/*

FROM MAIN :

    /*
	// Texture map init
	//g_nTexMapX = (((unsigned short)(g_depthMD.FullXRes()-1) / 512) + 1) * 512;
	//g_nTexMapY = (((unsigned short)(g_depthMD.FullYRes()-1) / 512) + 1) * 512;
	//g_pTexMap = (XnRGB24Pixel*)malloc(g_nTexMapX * g_nTexMapY * sizeof(XnRGB24Pixel));



	// OpenGL init
	//glutInit(&argc, argv);
	//glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	//glutInitWindowSize(GL_WIN_SIZE_X, GL_WIN_SIZE_Y);
	//glutCreateWindow ("OpenNI Simple Viewer");
	//glutFullScreen();
	//glutSetCursor(GLUT_CURSOR_NONE);

	//glutKeyboardFunc(glutKeyboard);
	//glutDisplayFunc(glutDisplay);
	//glutIdleFunc(glutIdle);

	//glDisable(GL_DEPTH_TEST);
	//glEnable(GL_TEXTURE_2D);

	// Per frame code is in glutDisplay

	//glutMainLoop();


    //avi = cvCreateVideoWriter("out.avi", CV_FOURCC('P','I','M','1') , 30, cvSize(640,480), TRUE);

    /*
    DepthMetaData depthMD;
    ImageMetaData imageMD;

    Mat depthshow;
    Mat show;


    END OF FROM MAIN
    */


    /*
void glutIdle (void)
{
	// Display the frame
	glutPostRedisplay();
}

void glutDisplay (void)
{
	XnStatus rc = XN_STATUS_OK;

	//avi = cvCreateVideoWriter("out.avi", CV_FOURCC('P','I','M','1'), 30, cvSize(640,480), TRUE);


    int i = 0;
	while ((rc = g_image.WaitAndUpdateData()) != XN_STATUS_EOF && (rc = g_depth.WaitAndUpdateData()) != XN_STATUS_EOF && i<140)
	{
        if (rc != XN_STATUS_OK)
        {
            printf("Read failed: %s\n", xnGetStatusString(rc));
            return;
        }

        g_depth.GetMetaData(g_depthMD);
        g_image.GetMetaData(g_imageMD);

        const XnDepthPixel* pDepth = g_depthMD.Data();
//        const XnUInt8* pImage = g_imageMD.Data();

//       unsigned int nImageScale = GL_WIN_SIZE_X / g_depthMD.FullXRes();

        cv::Mat colorArr[3];
        cv::Mat colorImage;
        const XnRGB24Pixel* pImageRow;
        const XnRGB24Pixel* pPixel;

        //imageGen.SetPixelFormat(XN_PIXEL_FORMAT_RGB24 );  //
        //xn::ImageGenerator imageGen;
        //imageGen.GetMetaData(imageMD);    //xn::ImageMetaData imageMD;
        pImageRow = g_imageMD.RGB24Data();

        colorArr[0] = cv::Mat(g_imageMD.YRes(),g_imageMD.XRes(),CV_8U);
        colorArr[1] = cv::Mat(g_imageMD.YRes(),g_imageMD.XRes(),CV_8U);
        colorArr[2] = cv::Mat(g_imageMD.YRes(),g_imageMD.XRes(),CV_8U);

        for (unsigned int y=0; y<g_imageMD.YRes(); y++) {
            pPixel = pImageRow;
            uchar* Bptr = colorArr[0].ptr<uchar>(y);
            uchar* Gptr = colorArr[1].ptr<uchar>(y);
            uchar* Rptr = colorArr[2].ptr<uchar>(y);
            for(unsigned int x=0;x<g_imageMD.XRes();++x , ++pPixel){
                Bptr[x] = pPixel->nBlue;
                Gptr[x] = pPixel->nGreen;
                Rptr[x] = pPixel->nRed;
            }
            pImageRow += g_imageMD.XRes();
        }
        cv::merge(colorArr,3,colorImage);

        IplImage iplImage = colorImage;
        if ( avi == NULL) {
            i=i+1;
            printf ( " dupa%d \n",i);
        }
        //cvWriteFrame (avi, &iplImage);

        cvShowImage("mainWin", &iplImage);
        //cvNamedWindow("123",1);
        cvWaitKey(33);           // wait 20 ms

        // Copied from SimpleViewer
        // Clear the OpenGL buffers
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Setup the OpenGL viewpoint
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0, GL_WIN_SIZE_X, GL_WIN_SIZE_Y, 0, -1.0, 1.0);

        // Calculate the accumulative histogram (the yellow display...)
        xnOSMemSet(g_pDepthHist, 0, MAX_DEPTH*sizeof(float));

        unsigned int nNumberOfPoints = 0;
        for (XnUInt y = 0; y < g_depthMD.YRes(); ++y)
        {
            for (XnUInt x = 0; x < g_depthMD.XRes(); ++x, ++pDepth)
            {
                if (*pDepth != 0)
                {
                    g_pDepthHist[*pDepth]++;
                    nNumberOfPoints++;
                }
            }
        }
        for (int nIndex=1; nIndex<MAX_DEPTH; nIndex++)
        {
            g_pDepthHist[nIndex] += g_pDepthHist[nIndex-1];
        }
        if (nNumberOfPoints)
        {
            for (int nIndex=1; nIndex<MAX_DEPTH; nIndex++)
            {
                g_pDepthHist[nIndex] = (unsigned int)(256 * (1.0f - (g_pDepthHist[nIndex] / nNumberOfPoints)));
            }
        }

        xnOSMemSet(g_pTexMap, 0, g_nTexMapX*g_nTexMapY*sizeof(XnRGB24Pixel));

        // check if we need to draw image frame to texture
        if (g_nViewState == DISPLAY_MODE_OVERLAY ||
            g_nViewState == DISPLAY_MODE_IMAGE)
        {
            const XnRGB24Pixel* pImageRow = g_imageMD.RGB24Data();
            XnRGB24Pixel* pTexRow = g_pTexMap + g_imageMD.YOffset() * g_nTexMapX;

            for (XnUInt y = 0; y < g_imageMD.YRes(); ++y)
            {
                const XnRGB24Pixel* pImage = pImageRow;
                XnRGB24Pixel* pTex = pTexRow + g_imageMD.XOffset();

                for (XnUInt x = 0; x < g_imageMD.XRes(); ++x, ++pImage, ++pTex)
                {
                    *pTex = *pImage;
                }

                pImageRow += g_imageMD.XRes();
                pTexRow += g_nTexMapX;
            }
        }

        // check if we need to draw depth frame to texture
        if (g_nViewState == DISPLAY_MODE_OVERLAY ||
            g_nViewState == DISPLAY_MODE_DEPTH)
        {
            const XnDepthPixel* pDepthRow = g_depthMD.Data();
            XnRGB24Pixel* pTexRow = g_pTexMap + g_depthMD.YOffset() * g_nTexMapX;

            for (XnUInt y = 0; y < g_depthMD.YRes(); ++y)
            {
                const XnDepthPixel* pDepth = pDepthRow;
                XnRGB24Pixel* pTex = pTexRow + g_depthMD.XOffset();

                for (XnUInt x = 0; x < g_depthMD.XRes(); ++x, ++pDepth, ++pTex)
                {
                    if (*pDepth != 0)
                    {
                        int nHistValue = g_pDepthHist[*pDepth];
                        pTex->nRed = nHistValue;
                        pTex->nGreen = nHistValue;
                        pTex->nBlue = 0;
                    }
                }

                pDepthRow += g_depthMD.XRes();
                pTexRow += g_nTexMapX;
            }
        }

        // Create the OpenGL texture map
        glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, g_nTexMapX, g_nTexMapY, 0, GL_RGB, GL_UNSIGNED_BYTE, g_pTexMap);

        // Display the OpenGL texture map
        glColor4f(1,1,1,1);

        glBegin(GL_QUADS);

        int nXRes = g_depthMD.FullXRes();
        int nYRes = g_depthMD.FullYRes();

        // upper left
        glTexCoord2f(0, 0);
        glVertex2f(0, 0);
        // upper right
        glTexCoord2f((float)nXRes/(float)g_nTexMapX, 0);
        glVertex2f(GL_WIN_SIZE_X, 0);
        // bottom right
        glTexCoord2f((float)nXRes/(float)g_nTexMapX, (float)nYRes/(float)g_nTexMapY);
        glVertex2f(GL_WIN_SIZE_X, GL_WIN_SIZE_Y);
        // bottom left
        glTexCoord2f(0, (float)nYRes/(float)g_nTexMapY);
        glVertex2f(0, GL_WIN_SIZE_Y);

        glEnd();

        // Swap the OpenGL display buffers
        glutSwapBuffers();
    }
    cvReleaseVideoWriter(&avi);
    //g_context.Shutdown();

}

void glutKeyboard (unsigned char key, int x, int y)
{
	switch (key)
	{
		case 27:
			exit (1);
		case '1':
			g_nViewState = DISPLAY_MODE_OVERLAY;
			g_depth.GetAlternativeViewPointCap().SetViewPoint(g_image);
			break;
		case '2':
			g_nViewState = DISPLAY_MODE_DEPTH;
			g_depth.GetAlternativeViewPointCap().ResetViewPoint();
			break;
		case '3':
			g_nViewState = DISPLAY_MODE_IMAGE;
			g_depth.GetAlternativeViewPointCap().ResetViewPoint();
			break;
		case 'm':
			g_context.SetGlobalMirror(!g_context.GetGlobalMirror());
			break;
	}
}

*/

