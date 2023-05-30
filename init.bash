mkdir datagen/data
mkdir datagen/infer_DB
mkdir datagen/infer_DB/infer_clips
mkdir datagen/infer_DB/infer_out
mkdir datagen/infer_DB/infer_pred

cd datagen/JAAD_DS
wget http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip
unzip JAAD_clips.zip
rm JAAD_clips.zip
