from config import args
from utils import load_data, build_vocab, gen_submission, gen_final_submission, eval_based_on_outputs
from model import Model

if __name__ == '__main__':
    if not args.pretrained:
        print('No pretrained model specified.')
        exit(0)
    build_vocab()

    if args.test_mode:
        dev_data = load_data('./data/test-data-processed.json')
        show_data = load_data('./data/show-data-processed.json')
    else:
        dev_data = load_data('./data/dev-data-processed.json')
        #show_data = load_data('./data/last_show_data.json')
        #show_data = load_data('./data/final_show_data_processed.json')
        #show_data = load_data('./data/first.json')
        #show_data = load_data('./data/second.json')
        #show_data = load_data('./data/four1.json')
        #show_data = load_data('./data/four2.json')
        #show_data = load_data('./data/Seven1.json')
        #show_data = load_data('./data/Seven2.json')
        #show_data = load_data('./data/Eight1.json')
        #show_data = load_data('./data/Eight2.json')
        #show_data = load_data('./data/401_1.json')
        show_data = load_data('./data/401_2.json')
        #show_data = load_data('./data/365.json')
        #show_data = load_data('./data/387_0.json')
        #show_data = load_data('./data/387_1.json')
    model_path_list = args.pretrained.split(',')
    for model_path in model_path_list:
        print('Load model from %s...' % model_path)
        args.pretrained = model_path
        model = Model(args)

        # evaluate on development dataset
        dev_acc = model.evaluate(dev_data)
        #print('dev accuracy: %f' % dev_acc)
        model.evaluate(show_data,debug=True,is_paint=True)
        # generate submission zip file for Codalab
        prediction = model.predict(dev_data)
        gen_submission(dev_data, prediction)

    gen_final_submission(dev_data)
    if not args.test_mode:
        eval_based_on_outputs('./answer.txt')
    model.evaluate(show_data,debug=True,is_paint=True)
