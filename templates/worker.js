// Web Worker for pixel processing
self.onmessage = function(e) {
    const { imageData, originalWidth, originalHeight, operation } = e.data;

    if (operation === 'extractRgb') {
        try {
            const pixelData = imageData.data;
            const totalPixels = originalWidth * originalHeight;
            const rgbData = new Uint8ClampedArray(totalPixels * 3);

            // Optimized RGB extraction
            for (let i = 0, j = 0; i < pixelData.length; i += 4, j += 3) {
                rgbData.set(pixelData.subarray(i, i + 3), j);
            }

            self.postMessage({
                success: true,
                rgbData: rgbData.buffer,
                operation: 'extractRgb'
            }, [rgbData.buffer]);
        } catch (error) {
            self.postMessage({
                success: false,
                error: error.message,
                operation: 'extractRgb'
            });
        }
    }

    if (operation === 'reconstructImage') {
        try {
            const processedRgbData = new Uint8ClampedArray(e.data.processedRgbData);
            const newImageData = new ImageData(originalWidth, originalHeight);
            const outputData = newImageData.data;

            // Optimized image reconstruction
            for (let i = 0, j = 0; i < outputData.length; i += 4, j += 3) {
                outputData[i] = processedRgbData[j];     // R
                outputData[i + 1] = processedRgbData[j + 1]; // G
                outputData[i + 2] = processedRgbData[j + 2]; // B
                outputData[i + 3] = 255; // Alpha set to opaque
            }

            self.postMessage({
                success: true,
                imageData: newImageData,
                operation: 'reconstructImage'
            });
        } catch (error) {
            self.postMessage({
                success: false,
                error: error.message,
                operation: 'reconstructImage'
            });
        }
    }
};