import math
import sys
from pathlib import Path
import easyocr
import cv2

from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


def computeRGBToGreyscale(r, g, b, image_width, image_height):
    
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for x in range(image_width):
            greyscale_pixel_array[i][x] = round(0.299 * r[i][x] + 0.587 * g[i][x] + 0.114 * b[i][x])
    return greyscale_pixel_array


def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    arr = createInitializedGreyscalePixelArray(image_width, image_height)

    for y in range(2, image_height-2):
            for x in range(2, image_width-2):

                total = 0.0
                for eta in [-2, -1, 0, 1, 2]:
                    for xi in [-2, -1, 0, 1, 2]:   # x axis
                        value = pixel_array[y+eta][x+xi]
                        total += value
                mean = total / 25

                variance = 0.0
                for eta in [-2, -1, 0, 1, 2]:
                    for xi in [-2, -1, 0, 1, 2]:   # x axis
                        value = (pixel_array[y+eta][x+xi] - mean) ** 2
                        variance += value
                variance /= 25
                variance = math.sqrt(variance)
                arr[y][x] = variance
    return arr


def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    quantized_greyscale = createInitializedGreyscalePixelArray(image_width, image_height)
    gmax, gmin = 255, 0
    fmax, fmin = 0, 255
    for row in pixel_array:
        fmax = max(max(row), fmax)
        fmin = min(min(row), fmin)

    if fmin != fmax:
        for i in range(image_height):
            for x in range(image_width):
                sout = round((pixel_array[i][x] - fmin) * ((gmax - gmin) / (fmax - fmin)) + gmin)
                # print(sout)
                if sout < gmin:
                    quantized_greyscale[i][x] = gmin
                elif sout > gmax:
                    quantized_greyscale[i][x] = gmax
                else:
                    quantized_greyscale[i][x] = sout
    return quantized_greyscale 


def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    for row in pixel_array:
        for i in range(len(row)):
            if row[i] < threshold_value:
                row[i] = 0
            else:
                row[i] = 255
    return pixel_array

def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    arr = createInitializedGreyscalePixelArray(image_width, image_height)

    for y in range(1, image_height-1):
        for x in range(1, image_width-1):

            grid = []
            for eta in [-1, 0, 1]:
                for xi in [-1, 0, 1]:   # x axis
                    value = pixel_array[y+eta][x+xi]
                    grid.append(value)

            if 0 not in grid:
                arr[y][x] = 1

    return arr


def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    arr = createInitializedGreyscalePixelArray(image_width, image_height)

    # border zero padding
    for y in range(image_height):
        pixel_array[y].insert(0, 0)
        pixel_array[y].append(0)
    pixel_array.insert(0, [0] * (image_width + 2))
    pixel_array.append([0] * (image_width + 2))


    for y in range(1, image_height+1):
        for x in range(1, image_width+1):

            grid = []
            for eta in [-1, 0, 1]:
                for xi in [-1, 0, 1]:   # x axis
                    value = pixel_array[y+eta][x+xi]
                    grid.append(value)

            if grid != [0] * 9:
                arr[y-1][x-1] = 1

    return arr


def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    class Queue:
        def __init__(self):
            self.items = []

        def isEmpty(self):
            return self.items == []

        def enqueue(self, item):
            self.items.insert(0,item)

        def dequeue(self):
            return self.items.pop()

        def size(self):
            return len(self.items)

    result = createInitializedGreyscalePixelArray(image_width, image_height)
    visited = createInitializedGreyscalePixelArray(image_width, image_height)
    visited_dict = {}
    label = 1
    label_count = {}
    
    # border zero padding
    for y in range(image_height):
        pixel_array[y].insert(0, 0)
        pixel_array[y].append(0)
    pixel_array.insert(0, [0] * (image_width + 2))
    pixel_array.append([0] * (image_width + 2))

    for y in range(1, image_height+1):
        for x in range(1, image_width+1):
            if pixel_array[y][x] != 0 and visited[y-1][x-1] != 1:

                visited[y-1][x-1] = 1
                q = Queue()
                q.enqueue((y, x))

                label_count[label] = 0

                while q.size() != 0:
                    py, px = q.dequeue()
                    
                    label_count[label] += 1
                    result[py-1][px-1] = label

                    if pixel_array[py][px-1] != 0 and visited[py-1][px-2] != 1:  # left
                        q.enqueue((py, px-1))
                        visited[py-1][px-2] = 1
            
                    if pixel_array[py][px+1] != 0 and visited[py-1][px] != 1:  # right
                        q.enqueue((py, px+1))
                        visited[py-1][px] = 1

                    if pixel_array[py-1][px] != 0 and visited[py-2][px-1] != 1:  # up
                        q.enqueue((py-1, px))
                        visited[py-2][px-1] = 1

                    if pixel_array[py+1][px] != 0 and visited[py][px-1] != 1:  # down
                        q.enqueue((py+1, px))
                        visited[py][px-1] = 1

                label += 1

    return result, label_count


def removeAllExceptComponent(pixel_array, image_width, image_height, component_val):
    for y in range(image_height):
        for x in range(image_width):
            if pixel_array[y][x] != component_val:
                pixel_array[y][x] = 0
    return pixel_array


def findBoundingBoxMinCoords(px_array, image_width, image_height, component):
    min_x, min_y = 99999999999, 99999999999
    for y in range(image_height):
        for x in range(image_width):
            if px_array[y][x] == component:
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
    return min_x, min_y


def findBoundingBoxMaxCoords(px_array, image_width, image_height, component):
    max_x, max_y = 0, 0
    for y in range(image_height):
        for x in range(image_width):
            if px_array[y][x] == component:
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y
    return max_x, max_y


def findLargestComponentAspectRatio(pixel_array, image_width, image_height, label_count):
    sorted_component_labels = sorted(label_count.items(), key=lambda x: x[1], reverse=True)

    for i in sorted_component_labels:
        min_x, min_y = findBoundingBoxMinCoords(pixel_array, image_width, image_height, i[0])
        max_x, max_y = findBoundingBoxMaxCoords(pixel_array, image_width, image_height, i[0])
        width, height = max_x - min_x, max_y - min_y
        if 1 <= width / height <= 5:  # aspect ratio check
            return i[0]     # component label


# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate3.png"
    # input_filename = "bikeplate3.png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(input_filename.replace(".png", "_output.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])


    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 3)

    # STUDENT IMPLEMENTATION here

    px_array_gray = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    axs1[0, 0].set_title('Greyscale Image')
    axs1[0, 0].imshow(px_array_gray, cmap='gray')

    px_array = computeStandardDeviationImage5x5(px_array_gray, image_width, image_height)
    px_array = scaleTo0And255AndQuantize(px_array, image_width, image_height)
    axs1[0, 1].set_title('Standard Deviation + Stretch')
    axs1[0, 1].imshow(px_array, cmap='gray')

    px_array = computeThresholdGE(px_array, 150, image_width, image_height)
    for i in range(4):
        px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)
    for i in range(4):
        px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)
    axs1[0, 2].set_title('Thresholding + Morphology Closing')
    axs1[0, 2].imshow(px_array, cmap='gray')

    px_array, components = computeConnectedComponentLabeling(px_array, image_width, image_height)
    # largest_object = max(components, key=components.get)
    largest_object = findLargestComponentAspectRatio(px_array, image_width, image_height, components)
    px_array = removeAllExceptComponent(px_array, image_width, image_height, largest_object)
    # axs1[1, 0].set_title('Largest Component')
    # axs1[1, 0].imshow(px_array, cmap='gray')

    bbox_min_x, bbox_min_y = findBoundingBoxMinCoords(px_array, image_width, image_height, largest_object)
    bbox_max_x, bbox_max_y = findBoundingBoxMaxCoords(px_array, image_width, image_height, largest_object)
    axs1[1, 0].set_title('Boundary Box of Largest Component')
    axs1[1, 0].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 0].add_patch(rect)


    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array_gray, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=3,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)


    # OCR to read license plate number / text
    plate_width, plate_height = bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y

    reader = easyocr.Reader(['en'])
    img = cv2.imread(input_filename)  # crop the image read by cv2 to the boundary box
    plate_img = img[bbox_min_y:bbox_max_y, bbox_min_x:bbox_max_x]
    result = reader.readtext(plate_img)
    plate_text = []  # arr of tuples with text, probability
    for value in result:
        plate_text.append((value[1], value[2] * 100))
    print(result)

    axs1[1, 2].set_title(f"Plate Text: {[text[0] for text in plate_text]}")
    axs1[1, 2].imshow(plate_img, cmap='gray')


    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()