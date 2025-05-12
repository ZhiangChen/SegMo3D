# Updates about Segmo3D
Several technics have been implemented to improve the algorithm efficiency. 

## 1. Refine images
In some cases when a point cloud is cropped to only associate part of the images, the Segmo3D algorithm processes every image and check their keyimages, without excluding those images not associated with the cropped point cloud. The first update is to use the keyimage associations, which include pixel-point associations, to exclude images not associated with the point cloud. When the images are refined, their corresponding keyimages are also refined accordingly. 


## 2. Parallel process
When processing each image, parallel computing is employed to handle multiple segmentation masks simultaneously. A key technical improvement is the use of shared memory for storing association variables. The parallel process significantly enhances efficiency, especially when the number of segmentation masks is large. 


# 3. Reduce point cloud size
When dealing with large point clouds, the program requires substantial memory and processing time. To address this, we can split a large point cloud into smaller chunks and process them sequentially. Although this sequential approach takes a similar amount of total processing time as handling the entire point cloud at once, it significantly reduces the memory required. Additionally, we can lower the point cloud density for the Segmo3D process to further improve efficiency.
