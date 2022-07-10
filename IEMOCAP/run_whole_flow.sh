
for i in {100..109}
do
	python3 svm.py --random_num "$i"
done

python3 avg_std.py --file_name uar.txt
python3 avg_std.py --file_name precision.txt
