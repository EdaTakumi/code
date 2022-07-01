import cv2 as cv
import glob

def MakeMovie(size, input_path, output_path, fps=5):
    pic_data = glob.glob(input_path)
    fourcc = cv.VideoWriter_fourcc('m','p','4','v')
    save = cv.VideoWriter(output_path, fourcc, fps, size)

    print('Now loading...')

    for i in range(len(pic_data)):
        img = pic_data[i]
        img = cv.imread(img)
        img=cv.resize(img, size)
        save.write(img)
    
    print('Completed!')

    save.release()


if __name__ == '__main__':
    img = cv.imread("C:\\dataset\\scene-flow\\MPI-Sintel\\MPI-Sintel-complete\\training\\final\\market_2\\frame_0001.png")
    height, width = img.shape[:2]
    size = (width, height)
    input_path = "C:\\dataset\\scene-flow\\MPI-Sintel\\MPI-Sintel-complete\\training\\final\\market_2\\*.png"
    output_path = "C:\\Users\\takumi_eda\\Desktop\\code\\market_2.mp4"
    MakeMovie(size, input_path, output_path)

