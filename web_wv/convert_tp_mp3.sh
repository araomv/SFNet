for i in *.wav;do
ffmpeg -i $i -ar 16000 -ac 2 -b:a 192k ${i%.*}.mp3;
done;
