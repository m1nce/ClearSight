#!/bin/bash

cd ../data

cp -r yolo_labels yolo_labels1
cp -r yolo_labels yolo_labels2
cp -r yolo_labels yolo_labels3
cp -r cityscapes cityscapes1 
cp -r aug_cityscapes aug_cityscapes1
cp -r aug_cityscapes aug_cityscapes2

# organize clear_cityscapes directory
mv yolo_labels cityscapes/
cd cityscapes
mv yolo_labels labels
mv labels/train/*/* labels/train/
mv labels/val/*/* labels/val/
mv labels/test/*/* labels/test/
mkdir images
mv train/*/* train/
mv val/*/* val/
mv test/*/* test/
rm -rf test/berlin
rm -rf train/hamburg
rm -rf train/stuttgart
rm -rf val/frankfurt
rm -rf val/lindau
mv train images/
mv val images/
mv test images/
rm -rf labels/train/hamburg
rm -rf labels/train/stuttgart
rm -rf labels/val/frankfurt
rm -rf labels/val/lindau
rm -rf labels/test/berlin
cd ..
mv cityscapes clear_cityscapes

# organize foggy_cityscapes directory
mv aug_cityscapes foggy_cityscapes
mv yolo_labels1 foggy_cityscapes
cd foggy_cityscapes
mv yolo_labels1 labels
mv labels/train/*/* labels/train/
mv labels/val/*/* labels/val/
mv labels/test/*/* labels/test/
cd test
rm -rf berlin_glaring
mv berlin_foggy berlin
cd ../val
rm -rf frankfurt_glaring
rm -rf lindau_glaring
mv frankfurt_foggy frankfurt
mv lindau_foggy lindau
cd ../train 
rm -rf hamburg_glaring
rm -rf stuttgart_glaring
mv hamburg_foggy hamburg
mv stuttgart_foggy stuttgart
cd ..
mkdir images
mv train/*/* train/
mv val/*/* val/
mv test/*/* test/
rm -rf test/berlin
rm -rf train/hamburg
rm -rf train/stuttgart
rm -rf val/frankfurt
rm -rf val/lindau
mv train images/
mv val images/
mv test images/
rm -rf labels/train/hamburg
rm -rf labels/train/stuttgart
rm -rf labels/val/frankfurt
rm -rf labels/val/lindau
rm -rf labels/test/berlin
cd ..

# organize glaring_cityscapes directory
mv aug_cityscapes1 glaring_cityscapes
mv yolo_labels2 glaring_cityscapes
cd glaring_cityscapes
mv yolo_labels2 labels
mv labels/train/*/* labels/train/
mv labels/val/*/* labels/val/
mv labels/test/*/* labels/test/
cd test
rm -rf berlin_foggy
mv berlin_glaring berlin
cd ../val
rm -rf frankfurt_foggy
rm -rf lindau_foggy
mv frankfurt_glaring frankfurt
mv lindau_glaring lindau
cd ../train 
rm -rf hamburg_foggy
rm -rf stuttgart_foggy
mv hamburg_glaring hamburg
mv stuttgart_glaring stuttgart
cd ..
mkdir images
mv train/*/* train/
mv val/*/* val/
mv test/*/* test/
rm -rf test/berlin
rm -rf train/hamburg
rm -rf train/stuttgart
rm -rf val/frankfurt
rm -rf val/lindau
mv train images/
mv val images/
mv test images/
rm -rf labels/train/hamburg
rm -rf labels/train/stuttgart
rm -rf labels/val/frankfurt
rm -rf labels/val/lindau
rm -rf labels/test/berlin
cd ..

# -------------------- WORK ON THIS, MAKE NEW DIRECTORY THAT HAS EVERYTHING --------------------------- #