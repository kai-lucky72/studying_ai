import cv2 as cv
import numpy as np


def filter_video(video_source=0, height=500, width=500):
    video = cv.VideoCapture(video_source)
    subtractor = cv.createBackgroundSubtractorMOG2(20, 50)

    while True:
        ret, frame = video.read()

        if ret:
            mask = subtractor.apply(frame)
            cv.imshow("mask", mask)

            if cv.waitKey(1) == ord('x'):
                break
        else:
            print("Failed to capture video stream.")
            break

    cv.destroyAllWindows()
    video.release()


def two_screen(height=500, width=500, video_source=0):
    camera = cv.VideoCapture(video_source)

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Failed to capture video stream.")
            break

        laplacian = cv.Laplacian(frame, cv.CV_64F)
        laplacian = np.uint8(laplacian)
        cv.imshow("laplacian", laplacian)

        edges = cv.Canny(frame, 100, 100)
        cv.imshow("Canny", edges)

        cv.resizeWindow("laplacian", width, height)
        cv.resizeWindow("Canny", width, height)

        if cv.waitKey(1) == ord('x'):
            break

    camera.release()
    cv.destroyAllWindows()


def process_image(image_path):
    img = cv.imread(image_path)
    if img is None:
        print(f"Image not found at {image_path}")
        return

    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    adaptive = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 81, 7)

    _, result = cv.threshold(img, 30, 200, cv.THRESH_BINARY)
    cv.imshow("adaptive", adaptive)
    cv.imshow("image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    path = r"C:\Users\user\Videos\SILVIZO - NIBIDO @PARODY.mp4"
    image_path = r"/projects/AI projects/AIproject/WIN_20240324_18_41_45_Pro.jpg"

    # Uncomment the function you want to run
    # filter_video(video_source=path)
    # two_screen(height=500, width=500)
    process_image(image_path)
