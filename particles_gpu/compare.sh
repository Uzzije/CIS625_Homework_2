./serial -o output-serial.txt
./gpu -o output-gpu.txt
diff output-serial.txt output-gpu.txt | wc
