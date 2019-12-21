mkdir images zips annotations
pushd images
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_0.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_2.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_3.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_4.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_5.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_6.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_7.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_8.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_9.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_b.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_e.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_d.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_f.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_1.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_a.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_c.tar.gz .
popd
pushd zips
find . -maxdepth 1 -type f -iname '*.tar.gz' -exec tar xzvf '{}' --directory ../images/ \;
popd
