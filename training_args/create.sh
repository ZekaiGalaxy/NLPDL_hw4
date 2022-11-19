for model in bert roberta scibert
do
    for dataset in res acl ag
        do
            for mode in train eval
                do 
                    touch "${mode}_${model}_$dataset.json"
                done
        done
done