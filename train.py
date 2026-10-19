from src.models.train_model import main, train_arg_parser

if __name__ == "__main__":
    args = train_arg_parser().parse_args()
    main(args.config, args.model_type, args.param)