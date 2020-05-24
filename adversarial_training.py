from net.generator import AlltagCopyCtxGenerator
import config
from utils import get_logger
from build_gan import Gan

cfg = config.cfg
log = get_logger(__name__)

def adversarial_training():
    inputs = read_inputs()
    model = Gan(cfg.rtm.UnkGeneratorCfg.model).to(device)
    trainer = GANTrainer(model)
    resume_trainer = config.cfg.resume_trainer
    if resume_trainer is not None:
        state_dict = torch.load(resume_trainer)
        trainer.load_state_dict(state_dict)
        if config.cfg.atk_validate:
            trainer.network.update()
            get_attack_score(trainer.network.gen)
            return
    try:
        trainer.train_by_batch_decoupled()
    except KeyboardInterrupt:
        obf, adv = model.get_trainers()
        obf_data_iter = obf.network.get_data_iter()[1](inputs.data_test)
        obf_meter = obf.validate(obf_data_iter)
        adv.network.relaxed_word_emb = False
        adv_data_iter = adv.network.get_data_iter()[1](inputs.data_test)
        adv_meter = adv.validate(adv_data_iter)
        uas, atk_acc = obf_meter.uas, adv_meter.atk_acc

        result = {
                "exp_name": config.cfg.exp_name,
                "exp_time": config.cfg.exp_time,
                "obf_meter": obf_meter,
                "adv_meter": adv_meter,
                "uas": uas,
                "acc": atk_acc
            }
        name = "{exp_name}-{exp_time}-{uas:.4f}-{acc:.4f}".format(**result)
        print(name)
        save_path = os.path.join("$ROOT/Project/data/result", name)
        torch.save(result, open(save_path, "wb"))
