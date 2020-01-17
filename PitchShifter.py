import numpy as np
import pydub
import matplotlib.pyplot as plt
import scipy.io.wavfile

j = complex(0,1)

def createFrames(x, hop, windowSize):

    # Find the max number of slices that can be obtained
    numberSlices = int(np.floor((len(x) - windowSize) / hop))

    # Truncate if needed to get only a integer number of hop
    x = x[0:(numberSlices*hop + windowSize)]

    # Create a matrix with time slices
    vectorFrames = np.zeros([int(np.floor(len(x) / hop)), windowSize])

    # Fill the matrix
    for index in range(0, numberSlices):
        vectorFrames[index, :] = x[(index*hop) : (index*hop + windowSize)]

    return vectorFrames, numberSlices

def fusionFrames(framesMatrix, hop):

    sizeMatrix = framesMatrix.shape

    # Get the number of frames
    numberFrames = int(sizeMatrix[0])

    # Get the size of each frame
    sizeFrames = int(sizeMatrix[1])

    # Define an empty vector to receive result
    vectorTime = np.zeros(int(numberFrames * hop - hop + sizeFrames))

    timeIndex = int(0)

    # Loop for each frame and overlap-add
    for index in range(0, numberFrames):
        vectorTime[timeIndex:(timeIndex+sizeFrames)] = vectorTime[timeIndex:(timeIndex+sizeFrames)] + framesMatrix[index, :]
        timeIndex = int(timeIndex + hop)
        
    return vectorTime

def pitchShift(inputArray, windowSize = 1024, hopSize = 256, step = 1):
    
    print("Initializing variables...")

    # Pitch scaling factor
    alpha = np.power(2, (step/12) )

    # Intermediate constants
    hopOut = round(alpha * hopSize)

    # Hanning window for overlap-add
    wn_tmp = np.hanning(windowSize*2 + 1)
    temp = windowSize*2 + 1
    wn = []
    for i in range(1, temp, 2):
        wn.append(wn_tmp[i])
    wn = np.array(wn)

    # Read the input array
    x = np.array(inputArray)

    x = np.append( np.zeros(hopSize*3), x )
    
    ######## Initialization ########

    # Create a frame matrix for the current input
    print("Creating Frames...")
    y, numberFramesInput = createFrames(x, hopSize, windowSize)

    # Create a frame matrix to receive processed frames
    numberFramesOutput = numberFramesInput
    outputy = np.zeros([numberFramesOutput, windowSize])

    # Initialize cumulative phase
    phaseCumulative = 0

    # Initialize previous frame phase
    previousPhase = 0

    print("Starting analysis (Pitch shift) ")
    ######## Analysis ########

    for index in range(0, numberFramesInput):
        # Get current frame to be processed
        currentFrame = y[index, :]
        
        # Window the frame
        currentFrameWindowed = currentFrame * wn / np.sqrt( (windowSize / hopSize) / 2)
        
        # Get the FFT
        currentFrameWindowedFFT = np.fft.fft(currentFrameWindowed)
        
        # Get the magnitude
        magFrame = np.abs(currentFrameWindowedFFT)
        
        # Get the angle
        phaseFrame = np.angle(currentFrameWindowedFFT)

        ######## Processing ########

        # Get the phase difference
        deltaPhi = phaseFrame - previousPhase
        previousPhase = phaseFrame
        
        # Remove the expected phase difference
        deltaPhiPrime = deltaPhi - hopSize * 2 * np.pi * np.array(range(0, (windowSize))) / windowSize
        
        # Map to -pi/pi range
        deltaPhiPrimeMod = np.mod(deltaPhiPrime + np.pi, 2 * np.pi) - np.pi
        
        # Get the true frequency
        trueFreq = 2 * np.pi * np.array(range(0, (windowSize))) / windowSize + deltaPhiPrimeMod/ hopSize

        # Get the final phase
        phaseCumulative = phaseCumulative + hopOut * trueFreq

        ######## Synthesis ########
    
        # Get the magnitude
        outputMag = magFrame
        
        # Produce output frame
        outputFrame = np.real( np.fft.ifft( outputMag * np.exp(j * phaseCumulative) ) )
        
        # Save frame that has been processed
        outputy[index, :] = outputFrame * wn / np.sqrt( (windowSize / hopOut) / 2)

    ####### Finalization #######

    # Overlap add in a vector
    print("Merging Frames...")
    outputTimeStretched = fusionFrames(outputy, hopOut)

    print("Interpolating the results...")
    # Resample with linear interpolation
    outputTime = np.interp( np.arange(0, len(outputTimeStretched) - 1, alpha) , np.arange(0, len(outputTimeStretched) ) , outputTimeStretched)

    # Return the result
    outputVector = outputTime

    return outputVector


inputFile = 'LauraBraniganSelfControl.wav'
sample_rate, sound_array = scipy.io.wavfile.read(inputFile)

print("Plotting Input Signal")
plt.title("Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
t = np.arange(0, len(sound_array))
plt.plot(t, sound_array)
plt.show()

out1 = pitchShift(inputArray = sound_array[:, 0])
out2 = pitchShift(inputArray = sound_array[:, 1])

out1 = out1.astype('int16')
out2 = out2.astype('int16')

out = np.stack( (out1, out2), axis=0 )
out = out.T

print("Plotting Output Signal")
plt.title("Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
t = np.arange(0, len(out))
plt.plot(t, out)
plt.show()


print("Writing to wav file")
scipy.io.wavfile.write("outSelfControl.wav", 44100, out)

