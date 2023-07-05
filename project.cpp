#include "project.hh"
#define HORIZ 2
#define DEG45 1
#define VERT 0
#define DEG135 3
#define PI CV_PI

void utcn::ip::Project::runLab() {
  int op;
  do {
    utcn::ip::Project::printMenu(LAB_MENU);
    std::cin >> op;
    switch (op) {
      case 0:
        break;
      case 1:
        testProject();
        break;

      default:
        std::cout << "Invalid selection" << std::endl;
    }
  } while (op != 0);
}

cv::Mat lineDetectionCustom(const cv::Mat& image, const cv::Mat& edgeImage,
                            int minLineLength = 1800, int numRhos = 180,
                            int numThetas = 180, int threshold = 1600) {
  int edgeHeight = edgeImage.rows;
  int edgeWidth = edgeImage.cols;
  double edgeHeightHalf = edgeHeight / 2.0;
  double edgeWidthHalf = edgeWidth / 2.0;
  double d = sqrt(pow(edgeHeight, 2) + pow(edgeWidth, 2));
  double dTheta = 180.0 / numThetas;
  double dRho = (2.0 * d) / numRhos;

  std::vector<double> thetas(numThetas);
  std::vector<double> rhos(numRhos);

  // used to quantize the angle and distance values in the Hough space.

  for (int i = 0; i < numThetas; i++) {
    thetas[i] = i * dTheta;
  }

  for (int i = 0; i < numRhos; i++) {
    rhos[i] = -d + i * dRho;
  }

  cv::Mat accumulator = cv::Mat::zeros(numRhos, numThetas, CV_32S);

  cv::Mat outputImage;
  cv::cvtColor(image, outputImage, cv::COLOR_GRAY2BGR);

  cv::Mat houghSpace;
  cv::cvtColor(edgeImage, houghSpace, cv::COLOR_GRAY2BGR);

  std::vector<cv::Point> edgePoints;
  cv::findNonZero(edgeImage, edgePoints);

  for (const cv::Point& edgePoint : edgePoints) {
    double x = edgePoint.x - edgeWidthHalf;
    double y = edgePoint.y - edgeHeightHalf;

    for (int t = 0; t < numThetas; t++) {
      double rhoValue = x * cos(thetas[t]) + y * sin(thetas[t]);
      int rhoIndex = static_cast<int>((rhoValue + d) / dRho);
      accumulator.at<int>(rhoIndex, t)++;
    }
  }

  for (int r = 0; r < numRhos; r++) {
    for (int t = 0; t < numThetas; t++) {
      int count = accumulator.at<int>(r, t);
      if (count > threshold) {
        double rho = rhos[r];
        double theta = thetas[t];
        double a = cos(theta);
        double b = sin(theta);
        double x0 = a * rho + edgeWidthHalf;
        double y0 = b * rho + edgeHeightHalf;
        double x1 = x0 - 1000 * (-b);
        double y1 = y0 - 1000 * (a);
        double x2 = x0 + 1000 * (-b);
        double y2 = y0 + 1000 * (a);
        double lineLength = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
        if (lineLength > minLineLength) {
          cv::line(houghSpace, cv::Point(x1, y1), cv::Point(x2, y2),
                   cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
          cv::line(outputImage, cv::Point(x1, y1), cv::Point(x2, y2),
                   cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }
      }
    }
  }

  // cv::Mat colormap;
  // cv::applyColorMap(accumulator, colormap, cv::COLORMAP_JET);

  cv::imshow("Hough Space Custom", houghSpace);
  cv::waitKey(0);

  cv::imshow("Detected Lines Custom Hough", outputImage);
  cv::waitKey(0);

  return accumulator;
}

void cannyAlgorithm(const cv::Mat& src, cv::Mat& dst) {
  int height = src.rows;
  int width = src.cols;

  // Gaussian filter
  cv::Mat Gx(height, width, CV_32S);  // gradient along x
  cv::Mat Gy(height, width, CV_32S);  // gradient along y
  cv::Mat G(height, width, CV_8UC1);
  G.setTo(0);
  cv::Mat Gmax(height, width, CV_8UC1);  // non maxima suppressed image
  Gmax.setTo(0);
  cv::Mat dir(height, width, CV_8UC1);

  // intensity gradient
  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
      Gx.at<int>(i, j) =
          (src.at<uchar>(i - 1, j + 1) + 2 * src.at<uchar>(i, j + 1) +
           src.at<uchar>(i + 1, j + 1)) -
          (src.at<uchar>(i - 1, j - 1) + 2 * src.at<uchar>(i, j - 1) +
           src.at<uchar>(i + 1, j - 1));
      Gy.at<int>(i, j) =
          (src.at<uchar>(i - 1, j - 1) + 2 * src.at<uchar>(i - 1, j) +
           src.at<uchar>(i - 1, j + 1)) -
          (src.at<uchar>(i + 1, j - 1) + 2 * src.at<uchar>(i + 1, j) +
           src.at<uchar>(i + 1, j + 1));
      G.at<uchar>(i, j) = sqrt(Gx.at<int>(i, j) * Gx.at<int>(i, j) +
                               Gy.at<int>(i, j) * Gy.at<int>(i, j)) /
                          (4 * sqrt(2));

      uchar dirn = 0;
      float dir_rad = atan2(Gy.at<int>(i, j),
                            Gx.at<int>(i, j));  // compute direction in radians
      if (fabs(dir_rad) < PI / 8 || fabs(dir_rad) > 7 * PI / 8)
        dirn = HORIZ;
      else if ((dir_rad > PI / 8 && dir_rad < 3 * PI / 8) ||
               (dir_rad < -5 * PI / 8 && dir_rad > -7 * PI / 8))
        dirn = DEG45;
      else if ((dir_rad > 5 * PI / 8 && dir_rad < 7 * PI / 8) ||
               (dir_rad < -PI / 8 && dir_rad > -3 * PI / 8))
        dirn = DEG135;
      else
        dirn = VERT;

      dir.at<uchar>(i, j) = dirn;
    }
  }

  // non maxima suppression // loop through all the image pixels excluding
  // border
  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
      switch (dir.at<uchar>(i, j)) {
        case HORIZ:
          if (G.at<uchar>(i, j) < G.at<uchar>(i, j + 1) ||
              G.at<uchar>(i, j) < G.at<uchar>(i, j - 1))
            Gmax.at<uchar>(i, j) = 0;
          else
            Gmax.at<uchar>(i, j) = G.at<uchar>(i, j);
          break;
        case VERT:
          if (G.at<uchar>(i, j) < G.at<uchar>(i - 1, j) ||
              G.at<uchar>(i, j) < G.at<uchar>(i + 1, j))
            Gmax.at<uchar>(i, j) = 0;
          else
            Gmax.at<uchar>(i, j) = G.at<uchar>(i, j);
          break;
        case DEG45:  // diagonal
          if (G.at<uchar>(i, j) < G.at<uchar>(i - 1, j + 1) ||
              G.at<uchar>(i, j) < G.at<uchar>(i + 1, j - 1))
            Gmax.at<uchar>(i, j) = 0;
          else
            Gmax.at<uchar>(i, j) = G.at<uchar>(i, j);
          break;
          break;
        case DEG135:
          if (G.at<uchar>(i, j) < G.at<uchar>(i - 1, j - 1) ||
              G.at<uchar>(i, j) < G.at<uchar>(i + 1, j + 1))
            Gmax.at<uchar>(i, j) = 0;
          else
            Gmax.at<uchar>(i, j) = G.at<uchar>(i, j);
          break;
          break;
      }
    }
  }

  // double thresholding
  int histogram[256];  // histogram of gradient values
  memset(histogram, 0, sizeof(histogram));

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      histogram[Gmax.at<uchar>(i, j)]++;  // compute histogram
    }
  }
  float p = 0.1;                                  // predefined threshold
  int noNonEdge =
      (1 - p) * (height * width - histogram[0]);  // number of non edge pixels

  int sum = 0;
  int TH = 0;
  for (int i = 1; i < 256; i++) {
    sum += histogram[i];
    if (sum > noNonEdge) {
      TH = i;
      break;
    }
  }
  int TL = 0.4 * TH;

  // hysteresis thresholding
  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
      if (Gmax.at<uchar>(i, j) > TH)
        // strong edge
        dst.at<uchar>(i, j) = 255;
      else
        // no edge
        if (Gmax.at<uchar>(i, j) < TL) dst.at<uchar>(i, j) = 0;
        // weak edge
        else
          dst.at<uchar>(i, j) = 128;
    }
  }

  // edge tracking by hysteresis
  int* Qi = new int[height * width];
  int* Qj = new int[height * width];

  int st = 0;
  int end = 0;

  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
      if (dst.at<uchar>(i, j) == 255) {  // strong edge
        st = end = 0;
        Qi[end] = i;
        Qj[end++] = j;
        while (st < end) {  // while queue is not empty
          int ic = Qi[st];
          int jc = Qj[st++];

          for (int in = ic - 1; in <= ic + 1; in++) {
            for (int jn = jc - 1; jn <= jc + 1; jn++) {
              if (dst.at<uchar>(in, jn) == 128) {  // weak edge
                dst.at<uchar>(in, jn) = 255;
                Qi[end] = in;
                Qj[end++] = jn;
              }
            }
          }
        }
      }
    }
  }

  delete[] Qi;
  delete[] Qj;

  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
      if (dst.at<uchar>(i, j) == 128)
        dst.at<uchar>(i, j) = 0;  // remove remaining weak edges
    }
  }
  cv::imshow("Output image", G);
  cv::imshow("Gradient", Gmax);
}

void utcn::ip::Project::testProject() {
  const std::string abs_image_path = fileutil::getSingleFileAbsPath();
  cv::Mat src;
  if (!abs_image_path.empty()) {
    src = cv::imread(abs_image_path, cv::IMREAD_GRAYSCALE);
  }

  int height = src.rows;
  int width = src.cols;

  cv::Mat colorSrc;
  cv::cvtColor(src, colorSrc, cv::COLOR_GRAY2BGR);

  if (!src.empty()) {
    // Step 1: Load the grayscale image
    cv::imshow("Source", src);
    cv::waitKey(0);

    // Canny();

    // Step 2: Apply the Canny algorithm
    cv::Mat edgeMap(height, width, CV_8UC1);
    cannyAlgorithm(src, edgeMap);
    cv::imshow("Canny", edgeMap);
    cv::waitKey(0);

    cv::Mat houghSpace;
    cv::cvtColor(edgeMap, houghSpace, cv::COLOR_GRAY2BGR);

    // Step 3: Hough Transform
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(edgeMap, lines, 1, CV_PI / 180, 170, 0, 0);

    // Step 4: Detect lines of significant length
    for (size_t i = 0; i < lines.size(); i++) {
      float rho = lines[i][0];
      float theta = lines[i][1];

      double a = cos(theta);
      double b = sin(theta);
      double x0 = a * rho;
      double y0 = b * rho;

      cv::Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
      cv::Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));

      cv::line(colorSrc, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
      cv::line(houghSpace, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    cv::imshow("Hough Space", houghSpace);
    cv::waitKey(0);

    // Show the final image with detected lines
    cv::imshow("Detected Lines", colorSrc);
    cv::waitKey(0);
    cv::Mat colorSrc2;
    cv::cvtColor(src, colorSrc2, cv::COLOR_GRAY2BGR);
    std::vector<cv::Vec4f> lines4;
    cv::HoughLinesP(edgeMap, lines4, 1, CV_PI / 180, 50, 150, 10);
    for (size_t i = 0; i < lines4.size(); i++) {
      cv::Vec4f l = lines4[i];
      cv::line(colorSrc2, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]),
               cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }
    cv::imshow("Detected Lines (in red) - Standard Hough Line Transform",
               colorSrc2);
    cv::waitKey(0);

    cv::Mat accumulator = lineDetectionCustom(src, edgeMap);
  }
}
