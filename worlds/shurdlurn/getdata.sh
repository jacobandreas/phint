if [ $1 == "pull" ]; then
scp sidaw@jamie:~/git/examples/* ./examples/
scp sidaw@jamie:~/git/logs/* ./logs/
elif [ $1 == "push" ]; then
    scp -r ./html sidaw@jamie:~/public_html/turk/
    scp -r ./logs sidaw@jamie:~/public_html/turk/
fi
