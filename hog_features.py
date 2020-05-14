import numpy as np
from PIL import Image
from scipy import ndimage




def hog_feature_vector(normalized_block):
    '''
        Parameters:-
            normalized_block: A numpy array of size (unnormalized_blocks.shape[0]-1,unnormalized_blocks.shape[1]-1,36) 
                              Represents all (36,1) vectors from normalized block
        Returns:-
            feature_vector: A numpy array of size (unnormalized_blocks.shape[0]-1*unnormalized_blocks.shape[1]-1*36).
                            Unrolled feature vector by concatenating all (36,1) vectors.
    '''     
    feature_vector = np.ravel(normalized_block)
    return feature_vector.reshape(-1,1)

def calculate_norm(mini_block):
    '''
        Parameters:-
            mini_block: A numpy array of size (2,2,9) .
        Returns:-
            normed_vector: (36,1) shaped normalized vector.
    '''
    
    unnormed_vector = np.ravel(mini_block)
    normed_value = np.linalg.norm(unnormed_vector)
    normed_vector = np.divide(unnormed_vector,normed_value)
    # print(unnormed_vector.shape)
    return normed_vector

def block_normalization(unnormalized_blocks):
    '''
        Parameters:-
            unnormalized_blocks: A numpy array of size ((magnitude_cell.shape[0]//8 , magnitude_cell.shape[1]//8,9)) .
                                 Stores the 9 bins of each 8x8 cell.
        Returns:-
            normalized_block: A numpy array of size ((unnormalized_blocks.shape[0]-1*unnormalized_blocks.shape[1]-1,36)) .
                              Normalized value consisting of two 8x8 cells at a time and rolling it over the image.
    '''
    
    x_dims = unnormalized_blocks.shape[0]-1
    y_dims = unnormalized_blocks.shape[1]-1
    normalized_block = np.zeros((x_dims*y_dims,36))
    cell_count = 0
    for i in range(unnormalized_blocks.shape[0]-1):
        for j in range(unnormalized_blocks.shape[1]-1):
            x = unnormalized_blocks[i:i+2,j:j+2,:]
            normalized_block[cell_count] = calculate_norm(unnormalized_blocks[i:i+2,j:j+2,:])
            cell_count+=1
    # print(normalized_block.shape)
    return normalized_block

def histogram_image(magnitude_cell,direction_cell):
    '''
        Parameters:-
            magnitude: A numpy array same size as that of gradients_x.
            direction: A numpy array same size as that of gradients_x.
        Returns:-
            total_cells: Numpy array of size ((magnitude_cell.shape[0]//8 , magnitude_cell.shape[1]//8,9)).
            This array stores the histogram of each 8x8 cell .
    '''

    total_cells = np.zeros((magnitude_cell.shape[0]//8 * magnitude_cell.shape[1]//8,9))
    cell_counter = 0
    for i in range(0,magnitude_cell.shape[0],8):
        for j in range(0,magnitude_cell.shape[1],8):
            total_cells[cell_counter] = histogram_cell(magnitude_cell[i:i+8,j:j+8],direction_cell[i:i+8,j:j+8])
            cell_counter+=1
        else:
            break
    total_cells = total_cells.reshape((magnitude_cell.shape[0]//8 , magnitude_cell.shape[1]//8,9))
    # print(total_cells)
    return total_cells

def histogram_cell(magnitude,direction,num_bins = 9):
    '''
        Parameters:-
            magnitude: A numpy array size 8x8 sliced from magnitude matrix.
                       Stores the magnitude of the gradients for each pixel
            direction: A numpy array size 8x8 sliced from direction matrix.
                       Stores the direction of the gradients for each pixel 
        Returns:-
            bins: Nummpy array of size 9. This array stores the histogram of a single 8x8 cell.
    '''
    magnitude = np.ravel(magnitude)
    direction = np.ravel(direction)
    bins = np.zeros(num_bins)
        
    direction = direction%180
        
    bin_pos_lower = np.ceil(np.divide(direction,20)).astype(int)
    bin_pos_upper = np.floor(np.divide(direction,20)).astype(int)

    bin_pos_lower = bin_pos_lower%num_bins
    bin_pos_upper = bin_pos_upper%num_bins

    lower_percentage = np.multiply(np.divide(np.abs(np.subtract(direction%160,20*bin_pos_lower)),20),magnitude)
    higher_percentage = magnitude - lower_percentage

    for i in range(len(bin_pos_lower)):
        bins[bin_pos_lower[i]]+=lower_percentage[i]
        bins[bin_pos_upper[i]]+=higher_percentage[i]
    # print(bins)
    return bins



def direction(gradients):
    '''
    Parameters:-
        gradients: A tuple consisting of (gradients_x,gradients_y)
                   gradient_x matrix stores the gradients in x direction.
                   gradient_y matrix stores the gradients in x direction.
    Returns:-
        direction_matrix: A numpy array same size as that of gradients_x.
                          Stores the direction of the gradients for each pixel
    '''
    gradient_x,gradient_y = gradients
    direction_matrix = np.arctan(gradient_y/gradient_x)*180/np.pi
    
    return direction_matrix

def magnitude(gradients):
    '''
    Parameters:-
        gradients:  A tuple consisting of (gradients_x,gradients_y)
                    gradient_x matrix stores the gradients in x direction.
                    gradient_y matrix stores the gradients in x direction.
    Returns:-
        magnitude_matrix: A numpy array same size as that of gradients_x
                          Stores the magnitude of the gradients for each pixel
    '''
    gradient_x,gradient_y = gradients
    magnitude_matrix = np.sqrt(gradient_x**2+gradient_y**2)

    return magnitude_matrix




def calculate_gradients(image_array,greyscale = False):
    '''
        Parameters:-
            image_array:Numpy array of the shape `image_size`
        Returns:-
            gradients: A tuple of (gradient_x,gradient_y) each of shape `(image_shape[0],image_shape[1])`
                       gradient_x matrix stores the gradients in x direction.
                       gradient_y matrix stores the gradients in x direction.
    '''
    if greyscale:
        filter_ = np.array([[-1,0,1]])
    else:
        filter_ = np.array([[[-1,0,1]]])

    gradient_x = ndimage.convolve(image_array,-filter_,mode='constant',cval=0) 
    gradient_y = ndimage.convolve(np.transpose(image_array),filter_,mode='constant',cval=0)
    
    gradient_y = -np.transpose(gradient_y)

    if greyscale:
        gradients = (gradient_x,gradient_y)

    gradients = np.max(gradient_x,axis = 2),np.max(gradient_y,axis = 2)

    return gradients


def preprocess(image,image_size = (64,128)):
    '''
        Parameters:-
            image:Unprocessed Image
            image_size: Default value as 128x256. 
        Returns:- 
            image:Numpy array of size  `image_size`.
                  PIL image converted to numpy array.
    '''
    image = image.resize(image_size)
    image = np.array(image)
    
    # print(image)
    return image


def load_image(image_path):
    '''
        Parameters:
            image_path:Path of the image
        Returns:
            image:Loaded image
    '''
    image = Image.open(image_path)
    return image

def calculate_hog_features(image_path):
    loaded_image = load_image(image_path)
    processed_image = preprocess(loaded_image)
    gradients = calculate_gradients(processed_image)    
    magnitude_matrix = magnitude(gradients)
    direction_matrix = direction(gradients)
    unnormalized_blocks =  histogram_image(magnitude_matrix,direction_matrix)
    normalized_block = block_normalization(unnormalized_blocks)
    feature_vector = hog_feature_vector(normalized_block).reshape(-1,1)
    return feature_vector



