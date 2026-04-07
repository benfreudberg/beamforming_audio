# beamforming_audio

## Introduction
The original goal of this project was to create a wearable microphone array with a camera that could pick out nearby faces and then use beamforming algorthims to collect sound coming only from the direction of those faces while filtering out all other extraneous sound. This targeted audio stream could then be fed into the user's noise canceling or noise isolating headphones. The headphones would block out everything as well as they can, and then this system would pipe back in just the sound the user wants to listen to - the people in their immediate vicinity. This proved to be a non-viable application because in order for the array to work well at the all of the frequencies common to human speech (a range from about 100Hz to 8kHz), the array would need to be too large to be wearable. However, the work is still interesting and maybe there is some other application for it.

Edit: I'm now looking into applying this to a bird-finding microphone array project, see the new section towards the end of this doc.

## Wearable Headphone Array Simulation
### Setup
The primary file in this repo is run_simulation.py. By editing the simulation setup code there, the user can place an arbitrary number of sound sources (under the hood, just a mono .wav file) at any point in 3d space. They can also place simulated microphones at arbitrary locations and use these to build their beamforming array. Ideally, these would be centered at the origin and follow some physically realizable positioning constraints. The simulation also includes a reference pair of "ears" centered at the origin. Finally, the user can specify the locations of targets that should be focused on by the beamforming algorithm.

In the example case implemented now, there are six sound sources. Two of these sources coincide with the locations of the two targets. There are also 22 microphones along a 2 meter long line with symmetric geometric spacing. The center two mics are 2 cm apart centered at the origin and the spacing between each subsequent mic is 1.28 times that of the previous spacing. The final step of running the simulation is to display a 3d plot showing the locations of all sound sources, microphones, and ears (targets aren't shown, but probably should be...). Though in this example implemenation, the z-value of every entity is 0.

### "Recording" Step
The first step is to simulate "recording" the environment with each mic. So each mic calculates its distance to each source and uses that with the speed of sound to calculate how much that source will be delayed by at that mic. Then it takes that source's .wav file, delays it by the appropriate amount, and adds it to its internal "recording". When this step is done, all microphones will have a mono "recording" of the environment including all the sound sources. However, each mic's recording will be slightly different from all of the others because of their differing positions. Because there are so many competing sources, it is hard to listen to this "recording" and pick out any individual sound source.

In addition to "recording" to each mic, the simulation also "records" to each "ear". This pair of tracks is then exported as a stereo .wav file and is meant to simulate what it would sound like as a person sitting in the middle of this environment. Because it is stereo, you can pick up the directionality of each source, so it is easier to follow than the single mono "recording" from one mic. But it is still a very busy noise environment.

### Audio Signal Processing and Beamforming
[This is an excellent resource I used to learn about beamforming.](https://pysdr.org/content/doa.html#) I won't go into detail about how beamforming works because it has been covered by others much better than I could. But I did want to touch on how it relates to audio as things are a bit different. Traditional beamforming is designed around RF applications where the signal of interest is at a single, known frequency. Like the 2.4, 5, or 6 GHz carrier frequencies that different versions of WiFi use. When a receiver listens to such a signal, it can be pre-tuned to only pick up the narrow band of frequencies it wants to listen to. Additionally, the data that it receives is often in a magnitude + phase format (real plus imaginary) in the frequency domain.

Beamforming works best when the array's receivers are spaced at 1/2 wavelength distance from each other - a number that is easy to calculate given the signal of interest's frequency. The different beamforming algorithms all aim to do one thing - they create a gain and phase delay to be applied to each receiver so that when all of the receivers' signals are added together, the result is data that appears to have been collected by a single, highly directional receiver pointed in the direction of the target.

Audio signals however, are recorded in the time domain with a singular amplitude (real number only) that measures the time-localized air pressure relative to the time-average air pressure at the brief snapshot that is the sampling period. Also, audio signals don't come at just one frequency. Human speaking ranges from about 100Hz to 8kHz. So we have two problems, 1) beamforming requires frequency-domain data but we have time-domain data, and 2) beamforming only works on a single frequency but we have a wide range of frequencies.

So we take the Short Time Fourier Transform. That is - we take a short window of our time domain data (3 ms in this simulator) and calculate the discrete time fourier transform on it for each mic. This gives us a magnitude and phase (real and imaginary) number for each of a number of frequency bins for each mic. The number of bins depends on the sampling rate of the original signal and the window size. We perform an entirely independent beamforming calculation for each frequency bin across all mics. This gives us a single magnitude and phase for each bin. Finally, we perform an inverse fourier transform which coalesces all of the frequency bin data back into a new 3 ms time-domain audio signal. Then we slide forward to the next 3ms window (actually, we slide forward by half the window size so that there is some overlap - I won't go into the reasons here).

When we have finished, we will have taken all of our mics' recordings and used them to create a single audio track that should contain un-distorted signal from the target direction, and as little as possible from every other direction.

#### Array Size and Spacing Considerations
As mentioned before, an ideal array should have receivers spaced 1/2 wavelength apart. For the upper end of the range of human speech, that is about 2.1 cm. For the lower end, it is 1.7 meters. Take a look at the [When d is not lambda/2](https://pysdr.org/content/doa.html#when-d-is-not-2) section of the resource I linked earlier for what happens if you have poor spacing. For our application, it is fairly easy to meet the requirements for the upper end of the speech frequencies. But for the lower end to have any sort of directionality, we need a very large array. This is why making a wearable array for speech audio is impractical. You can play with the array size and listen to the output and see that as you make the array smaller and smaller, the lower frequency portions of the undesired signals creep into the final output more and more. You could implement a high-pass filter to stop this, but you would also be cutting the lower frequencies from your target signal and it would sound distorted.

### Simulation Output
The final output of the simulation will be 7 audio files (3 plus 2 for each target). One is a the mono recording from a single microphone just as a reference. raw_ambient.wav is the stereo simulation from the "ears" recordings that is what it would sound like if you were in the simulated environment. Then each target gets _DS mono file and an _MVDR mono file. Delay and Sum is the simplest beamforming algorithm that just points in the right direction with no other considerations. Minimum Variance Distortionless Response is a more advanced beamforming algorithm that performs an optimization that aims to minimize the total power of the signal while maintaining a constraint that the gain in the direction of the target is 1. Finally, targets_applied_to_ears.wav is a stereo audio file that takes the _MVDR target tracks and re-applies them to their locations in the simulated environment and casts them to the "ears". This is what you would hear if you were in the simulated environment with headphones that were able to perfectly cancel out all sound and then just pipe in the beamformed target signals.

## Bird Finding Array Simulation
### Setup and Recording
The setup and recording steps are the same as the headphone workflow: place sound sources at known locations, position microphones, and simulate each microphone's recording of the acoustic environment with appropriate delays and attenuations based on distance.

The bird-finding application operates at 48 kHz (higher sample rate for better spatial resolution at bird-call frequencies) and has different goals from the headphone application and requries a different workflow.

### Approach Overview
Rather than pre-specifying target locations, the bird-finder dynamically discovers sound sources in real-time. The core pipeline operates as follows:

1. **Rolling FFT Analysis**: A reference microphone records the acoustic scene. A rolling 4-second window (updated every 100 ms) is Fourier-transformed to produce a sequence of magnitude spectra.

2. **Peak Frequency Tracking**: For each FFT frame, the top ~20 local-maxima frequencies are extracted. A continuity tracker maintains coherent peaks across frames, with a slight bias towards higher frequencies because those will give us better locating precision.

3. **Far-Field Localization**: For each active frequency peak, a direction-of-arrival (DOA) algorithm determines the azimuth and elevation of that frequency. Because many frequencies often come from the same physical source, nearby DOAs are clustered to yield distinct source estimates.

4. **Visualization and Auralization**: Source directions are plotted as an animated polar/elevation map, showing sources appearing, disappearing, and moving. The ambient stereo ear-track is exported alongside this visualization to provide auditory context.

5. **Per-Source Beamforming**: Once sources are localized, beamforming targets are created at each source location to extract isolated audio streams for each sound source, similar to the headphone workflow. These "source" audio tracks can be shared with the Merlin Bird ID app so that we can label each source with a type of bird.

## Sample Audio Sources
Some of the audio files used for the simulation stored under input_audio_files I created myself. Some were sourced from https://www.youtube.com/watch?v=NTiDNF0ofNk and some from https://pixabay.com/sound-effects/food-hall-49662/. Bird recordings are from https://xeno-canto.org/.
