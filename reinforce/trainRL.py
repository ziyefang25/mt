import os
import argparse
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from HierarchicalLSTM import RLModel

from HierarchicalLatentModel import HierarchicalLatentModel
from dataset import TextDataset
from dataloader import MyDataLoader
from trainerRl import Trainer
from utils import get_pretrained_weights, train_validation_test_split
from vocabulary import Vocabulary


def train(config, device):
    vocab_word_to_idx = Vocabulary(config.data_dir, config.vocab_file).word_to_idx
    vocab = list(vocab_word_to_idx.keys())
    dataset = TextDataset(config.train_files, config.label_file, config.vocab_file, config.data_dir)
    dataloader = MyDataLoader(dataset, config.batch_size)
    model = RLModel(vocab_size=len(vocab)).to(device)
    optimizer = Adam(model.parameters(), lr=config.lr)
    criterion = nn.NLLLoss(reduction='sum').to(device) # option 1

    # NOTE MODIFICATION (EMBEDDING)
    if config.pretrain:
        weights = get_pretrained_weights(config.output_dir, vocab, config.embed_dim, device)
        model.encoder.init_embeddings(weights)
    model.encoder.freeze_embeddings(config.freeze)

    trainer = Trainer(config, model, optimizer, criterion, dataloader)
    trainer.train()


if __name__ == '__main__':
    train_files = ['patient533.txt', 'patient621.txt', 'patient884.txt', 'patient435.txt', 'patient534.txt', 'patient589.txt', 'patient609.txt', 'patient936.txt', 'patient556.txt', 'patient30.txt', 'patient947.txt', 'patient75.txt', 'patient189.txt', 'patient498.txt', 'patient973.txt', 'patient678.txt', 'patient34.txt', 'patient489.txt', 'patient524.txt', 'patient785.txt', 'patient428.txt', 'patient143.txt', 'patient5.txt', 'patient920.txt', 'patient546.txt', 'patient518.txt', 'patient690.txt', 'patient51.txt', 'patient263.txt', 'patient324.txt', 'patient831.txt', 'patient888.txt', 'patient847.txt', 'patient774.txt', 'patient827.txt', 'patient31.txt', 'patient513.txt', 'patient243.txt', 'patient626.txt', 'patient914.txt', 'patient542.txt', 'patient793.txt', 'patient350.txt', 'patient580.txt', 'patient531.txt', 'patient840.txt', 'patient57.txt', 'patient526.txt', 'patient122.txt', 'patient142.txt', 'patient979.txt', 'patient642.txt', 'patient566.txt', 'patient450.txt', 'patient488.txt', 'patient545.txt', 'patient733.txt', 'patient353.txt', 'patient81.txt', 'patient26.txt', 'patient216.txt', 'patient215.txt', 'patient681.txt', 'patient275.txt', 'patient197.txt', 'patient895.txt', 'patient700.txt', 'patient615.txt', 'patient413.txt', 'patient843.txt', 'patient128.txt', 'patient695.txt', 'patient396.txt', 'patient617.txt', 'patient800.txt', 'patient359.txt', 'patient160.txt', 'patient841.txt', 'patient902.txt', 'patient758.txt', 'patient158.txt', 'patient794.txt', 'patient604.txt', 'patient405.txt', 'patient753.txt', 'patient925.txt', 'patient969.txt', 'patient293.txt', 'patient612.txt', 'patient671.txt', 'patient613.txt', 'patient912.txt', 'patient541.txt', 'patient227.txt', 'patient801.txt', 'patient471.txt', 'patient382.txt', 'patient562.txt', 'patient469.txt', 'patient957.txt', 'patient19.txt', 'patient574.txt', 'patient451.txt', 'patient561.txt', 'patient538.txt', 'patient481.txt', 'patient571.txt', 'patient68.txt', 'patient463.txt', 'patient606.txt', 'patient573.txt', 'patient301.txt', 'patient367.txt', 'patient867.txt', 'patient846.txt', 'patient598.txt', 'patient146.txt', 'patient323.txt', 'patient673.txt', 'patient317.txt', 'patient103.txt', 'patient544.txt', 'patient337.txt', 'patient41.txt', 'patient905.txt', 'patient997.txt', 'patient903.txt', 'patient55.txt', 'patient778.txt', 'patient223.txt', 'patient253.txt', 'patient387.txt', 'patient462.txt', 'patient436.txt', 'patient296.txt', 'patient510.txt', 'patient336.txt', 'patient576.txt', 'patient235.txt', 'patient500.txt', 'patient375.txt', 'patient988.txt', 'patient238.txt', 'patient485.txt', 'patient281.txt', 'patient910.txt', 'patient516.txt', 'patient737.txt', 'patient607.txt', 'patient169.txt', 'patient712.txt', 'patient876.txt', 'patient848.txt', 'patient971.txt', 'patient493.txt', 'patient480.txt', 'patient104.txt', 'patient782.txt', 'patient35.txt', 'patient819.txt', 'patient529.txt', 'patient732.txt', 'patient608.txt', 'patient202.txt', 'patient173.txt', 'patient389.txt', 'patient114.txt', 'patient985.txt', 'patient713.txt', 'patient972.txt', 'patient788.txt', 'patient454.txt', 'patient39.txt', 'patient663.txt', 'patient875.txt', 'patient810.txt', 'patient476.txt', 'patient141.txt', 'patient48.txt', 'patient486.txt', 'patient157.txt', 'patient171.txt', 'patient306.txt', 'patient898.txt', 'patient780.txt', 'patient555.txt', 'patient726.txt', 'patient491.txt', 'patient294.txt', 'patient811.txt', 'patient954.txt', 'patient151.txt', 'patient503.txt', 'patient634.txt', 'patient127.txt', 'patient962.txt', 'patient775.txt', 'patient120.txt', 'patient791.txt', 'patient340.txt', 'patient125.txt', 'patient887.txt', 'patient672.txt', 'patient817.txt', 'patient383.txt', 'patient90.txt', 'patient639.txt', 'patient764.txt', 'patient750.txt', 'patient798.txt', 'patient453.txt', 'patient702.txt', 'patient865.txt', 'patient138.txt', 'patient627.txt', 'patient675.txt', 'patient619.txt', 'patient512.txt', 'patient384.txt', 'patient565.txt', 'patient174.txt', 'patient449.txt', 'patient360.txt', 'patient666.txt', 'patient100.txt', 'patient868.txt', 'patient362.txt', 'patient792.txt', 'patient853.txt', 'patient442.txt', 'patient192.txt', 'patient892.txt', 'patient603.txt', 'patient664.txt', 'patient472.txt', 'patient224.txt', 'patient659.txt', 'patient763.txt', 'patient683.txt', 'patient416.txt', 'patient551.txt', 'patient364.txt', 'patient11.txt', 'patient431.txt', 'patient768.txt', 'patient62.txt', 'patient282.txt', 'patient877.txt', 'patient734.txt', 'patient536.txt', 'patient287.txt', 'patient740.txt', 'patient283.txt', 'patient108.txt', 'patient945.txt', 'patient581.txt', 'patient92.txt', 'patient244.txt', 'patient302.txt', 'patient177.txt', 'patient797.txt', 'patient255.txt', 'patient107.txt', 'patient835.txt', 'patient857.txt', 'patient28.txt', 'patient946.txt', 'patient426.txt', 'patient36.txt', 'patient728.txt', 'patient466.txt', 'patient504.txt', 'patient958.txt', 'patient611.txt', 'patient990.txt', 'patient719.txt', 'patient934.txt', 'patient240.txt', 'patient300.txt', 'patient265.txt', 'patient736.txt', 'patient521.txt', 'patient325.txt', 'patient577.txt', 'patient864.txt', 'patient403.txt', 'patient585.txt', 'patient145.txt', 'patient137.txt', 'patient218.txt', 'patient858.txt', 'patient303.txt', 'patient132.txt', 'patient697.txt', 'patient989.txt', 'patient411.txt', 'patient496.txt', 'patient756.txt', 'patient167.txt', 'patient286.txt', 'patient200.txt', 'patient869.txt', 'patient520.txt', 'patient658.txt', 'patient660.txt', 'patient264.txt', 'patient342.txt', 'patient440.txt', 'patient987.txt', 'patient807.txt', 'patient680.txt', 'patient727.txt', 'patient597.txt', 'patient928.txt', 'patient870.txt', 'patient809.txt', 'patient600.txt', 'patient121.txt', 'patient790.txt', 'patient575.txt', 'patient906.txt', 'patient379.txt', 'patient522.txt', 'patient441.txt', 'patient539.txt', 'patient249.txt', 'patient452.txt', 'patient977.txt', 'patient636.txt', 'patient795.txt', 'patient369.txt', 'patient918.txt', 'patient591.txt', 'patient67.txt', 'patient620.txt', 'patient421.txt', 'patient891.txt', 'patient464.txt', 'patient511.txt', 'patient818.txt', 'patient643.txt', 'patient277.txt', 'patient578.txt', 'patient953.txt', 'patient772.txt', 'patient894.txt', 'patient461.txt', 'patient631.txt', 'patient558.txt', 'patient148.txt', 'patient444.txt', 'patient670.txt', 'patient54.txt', 'patient380.txt', 'patient71.txt', 'patient916.txt', 'patient228.txt', 'patient596.txt', 'patient2.txt', 'patient260.txt', 'patient273.txt', 'patient656.txt', 'patient84.txt', 'patient986.txt', 'patient219.txt', 'patient149.txt', 'patient548.txt', 'patient590.txt', 'patient651.txt', 'patient290.txt', 'patient424.txt', 'patient940.txt', 'patient708.txt', 'patient295.txt', 'patient880.txt', 'patient706.txt', 'patient743.txt', 'patient633.txt', 'patient823.txt', 'patient161.txt', 'patient455.txt', 'patient217.txt', 'patient796.txt', 'patient992.txt', 'patient731.txt', 'patient376.txt', 'patient229.txt', 'patient996.txt', 'patient622.txt', 'patient201.txt', 'patient457.txt', 'patient289.txt', 'patient941.txt', 'patient170.txt', 'patient478.txt', 'patient347.txt', 'patient3.txt', 'patient960.txt', 'patient714.txt', 'patient836.txt', 'patient881.txt', 'patient930.txt', 'patient995.txt', 'patient963.txt', 'patient77.txt', 'patient175.txt', 'patient351.txt', 'patient50.txt', 'patient966.txt', 'patient515.txt', 'patient316.txt', 'patient646.txt', 'patient226.txt', 'patient102.txt', 'patient339.txt', 'patient297.txt', 'patient432.txt', 'patient929.txt', 'patient859.txt', 'patient221.txt', 'patient230.txt', 'patient897.txt', 'patient438.txt', 'patient422.txt', 'patient554.txt', 'patient40.txt', 'patient180.txt', 'patient400.txt', 'patient22.txt', 'patient231.txt', 'patient70.txt', 'patient759.txt', 'patient147.txt', 'patient234.txt', 'patient582.txt', 'patient832.txt', 'patient694.txt', 'patient517.txt', 'patient419.txt', 'patient305.txt', 'patient194.txt', 'patient911.txt', 'patient118.txt', 'patient395.txt', 'patient610.txt', 'patient696.txt', 'patient595.txt', 'patient569.txt', 'patient408.txt', 'patient744.txt', 'patient805.txt', 'patient924.txt', 'patient433.txt', 'patient629.txt', 'patient65.txt', 'patient116.txt', 'patient60.txt', 'patient552.txt', 'patient85.txt', 'patient93.txt', 'patient693.txt', 'patient952.txt', 'patient970.txt', 'patient649.txt', 'patient115.txt', 'patient602.txt', 'patient165.txt', 'patient879.txt', 'patient593.txt', 'patient247.txt', 'patient653.txt', 'patient98.txt', 'patient14.txt', 'patient209.txt', 'patient650.txt', 'patient644.txt', 'patient742.txt', 'patient856.txt', 'patient749.txt', 'patient984.txt', 'patient722.txt', 'patient484.txt', 'patient499.txt', 'patient685.txt', 'patient900.txt', 'patient586.txt', 'patient777.txt', 'patient179.txt', 'patient961.txt', 'patient628.txt', 'patient261.txt', 'patient242.txt', 'patient106.txt', 'patient816.txt', 'patient20.txt', 'patient204.txt', 'patient724.txt', 'patient519.txt', 'patient269.txt', 'patient975.txt', 'patient205.txt', 'patient38.txt', 'patient331.txt', 'patient594.txt', 'patient710.txt', 'patient882.txt', 'patient682.txt', 'patient948.txt', 'patient79.txt', 'patient357.txt', 'patient965.txt', 'patient701.txt', 'patient397.txt', 'patient709.txt', 'patient530.txt', 'patient344.txt', 'patient371.txt', 'patient720.txt', 'patient826.txt', 'patient506.txt', 'patient803.txt', 'patient502.txt', 'patient705.txt', 'patient746.txt', 'patient314.txt', 'patient6.txt', 'patient632.txt', 'patient213.txt', 'patient124.txt', 'patient913.txt', 'patient950.txt', 'patient755.txt', 'patient153.txt', 'patient921.txt', 'patient8.txt', 'patient616.txt', 'patient386.txt', 'patient630.txt', 'patient241.txt', 'patient346.txt', 'patient272.txt', 'patient190.txt', 'patient655.txt', 'patient490.txt', 'patient711.txt', 'patient896.txt', 'patient765.txt', 'patient983.txt', 'patient654.txt', 'patient624.txt', 'patient0.txt', 'patient188.txt', 'patient373.txt', 'patient618.txt', 'patient187.txt', 'patient313.txt', 'patient196.txt', 'patient625.txt', 'patient684.txt', 'patient477.txt', 'patient563.txt', 'patient505.txt', 'patient532.txt', 'patient980.txt', 'patient806.txt', 'patient248.txt', 'patient356.txt', 'patient232.txt', 'patient784.txt', 'patient9.txt', 'patient579.txt', 'patient915.txt', 'patient76.txt', 'patient89.txt', 'patient378.txt', 'patient757.txt', 'patient96.txt', 'patient254.txt', 'patient259.txt', 'patient739.txt', 'patient198.txt', 'patient415.txt', 'patient155.txt', 'patient414.txt', 'patient766.txt', 'patient648.txt', 'patient112.txt', 'patient645.txt', 'patient674.txt', 'patient623.txt', 'patient52.txt', 'patient417.txt', 'patient267.txt', 'patient967.txt', 'patient186.txt', 'patient368.txt', 'patient686.txt', 'patient923.txt', 'patient691.txt', 'patient822.txt', 'patient21.txt', 'patient251.txt', 'patient815.txt', 'patient328.txt', 'patient721.txt', 'patient72.txt', 'patient37.txt', 'patient358.txt', 'patient86.txt', 'patient456.txt', 'patient17.txt', 'patient97.txt', 'patient341.txt', 'patient837.txt', 'patient704.txt', 'patient998.txt', 'patient268.txt', 'patient528.txt', 'patient74.txt', 'patient25.txt', 'patient601.txt', 'patient447.txt', 'patient699.txt', 'patient874.txt', 'patient163.txt', 'patient474.txt', 'patient641.txt', 'patient560.txt', 'patient439.txt', 'patient978.txt', 'patient182.txt', 'patient587.txt', 'patient1.txt', 'patient707.txt', 'patient252.txt', 'patient322.txt', 'patient284.txt', 'patient662.txt', 'patient233.txt', 'patient821.txt', 'patient312.txt', 'patient139.txt', 'patient220.txt', 'patient871.txt', 'patient374.txt', 'patient547.txt', 'patient327.txt', 'patient354.txt', 'patient448.txt', 'patient183.txt', 'patient825.txt', 'patient136.txt', 'patient338.txt', 'patient885.txt', 'patient257.txt', 'patient459.txt', 'patient7.txt', 'patient458.txt', 'patient83.txt', 'patient525.txt', 'patient291.txt', 'patient676.txt', 'patient130.txt', 'patient412.txt', 'patient638.txt', 'patient66.txt', 'patient318.txt', 'patient907.txt', 'patient443.txt', 'patient540.txt', 'patient492.txt', 'patient348.txt', 'patient391.txt', 'patient814.txt', 'patient942.txt', 'patient184.txt', 'patient951.txt', 'patient334.txt', 'patient860.txt', 'patient352.txt', 'patient767.txt', 'patient390.txt', 'patient460.txt', 'patient483.txt', 'patient320.txt', 'patient677.txt', 'patient741.txt', 'patient214.txt', 'patient110.txt', 'patient751.txt', 'patient468.txt', 'patient119.txt', 'patient668.txt', 'patient349.txt', 'patient804.txt', 'patient80.txt', 'patient976.txt', 'patient592.txt', 'patient937.txt', 'patient279.txt', 'patient494.txt', 'patient117.txt', 'patient445.txt', 'patient429.txt', 'patient813.txt', 'patient326.txt', 'patient844.txt', 'patient550.txt', 'patient370.txt', 'patient523.txt', 'patient994.txt', 'patient64.txt', 'patient802.txt', 'patient716.txt', 'patient434.txt', 'patient385.txt', 'patient588.txt', 'patient850.txt', 'patient839.txt', 'patient210.txt', 'patient166.txt', 'patient935.txt', 'patient467.txt', 'patient398.txt', 'patient872.txt', 'patient808.txt', 'patient407.txt', 'patient667.txt', 'patient365.txt', 'patient61.txt', 'patient917.txt', 'patient570.txt', 'patient69.txt', 'patient640.txt', 'patient129.txt', 'patient24.txt', 'patient939.txt', 'patient760.txt', 'patient497.txt', 'patient725.txt', 'patient308.txt', 'patient237.txt', 'patient266.txt', 'patient514.txt', 'patient943.txt', 'patient661.txt', 'patient271.txt', 'patient679.txt', 'patient723.txt', 'patient854.txt', 'patient932.txt', 'patient851.txt', 'patient394.txt', 'patient280.txt', 'patient820.txt', 'patient329.txt', 'patient225.txt', 'patient256.txt', 'patient824.txt', 'patient156.txt', 'patient715.txt', 'patient315.txt', 'patient345.txt', 'patient307.txt', 'patient133.txt', 'patient101.txt', 'patient298.txt', 'patient999.txt', 'patient49.txt', 'patient113.txt', 'patient111.txt', 'patient527.txt', 'patient425.txt', 'patient401.txt', 'patient150.txt', 'patient335.txt', 'patient861.txt', 'patient657.txt', 'patient393.txt', 'patient465.txt', 'patient377.txt', 'patient537.txt', 'patient949.txt', 'patient735.txt', 'patient191.txt', 'patient185.txt', 'patient203.txt', 'patient27.txt', 'patient392.txt', 'patient509.txt', 'patient919.txt', 'patient99.txt', 'patient53.txt', 'patient583.txt', 'patient890.txt', 'patient799.txt', 'patient482.txt', 'patient564.txt', 'patient964.txt', 'patient131.txt']
    validation_files = ['patient361.txt', 'patient199.txt', 'patient838.txt', 'patient94.txt', 'patient834.txt', 'patient501.txt', 'patient786.txt', 'patient495.txt', 'patient381.txt', 'patient873.txt', 'patient776.txt', 'patient404.txt', 'patient446.txt', 'patient761.txt', 'patient159.txt', 'patient330.txt', 'patient164.txt', 'patient207.txt', 'patient842.txt', 'patient13.txt', 'patient886.txt', 'patient635.txt', 'patient304.txt', 'patient508.txt', 'patient614.txt', 'patient647.txt', 'patient752.txt', 'patient270.txt', 'patient770.txt', 'patient652.txt', 'patient762.txt', 'patient479.txt', 'patient878.txt', 'patient599.txt', 'patient409.txt', 'patient63.txt', 'patient829.txt', 'patient355.txt', 'patient47.txt', 'patient208.txt', 'patient410.txt', 'patient669.txt', 'patient278.txt', 'patient549.txt', 'patient926.txt', 'patient559.txt', 'patient955.txt', 'patient363.txt', 'patient718.txt', 'patient862.txt', 'patient535.txt', 'patient239.txt', 'patient319.txt', 'patient91.txt', 'patient783.txt', 'patient418.txt', 'patient4.txt', 'patient78.txt', 'patient10.txt', 'patient250.txt', 'patient553.txt', 'patient812.txt', 'patient211.txt', 'patient745.txt', 'patient333.txt', 'patient698.txt', 'patient274.txt', 'patient245.txt', 'patient938.txt', 'patient754.txt', 'patient543.txt', 'patient931.txt', 'patient828.txt', 'patient993.txt', 'patient178.txt', 'patient866.txt', 'patient849.txt', 'patient852.txt', 'patient584.txt', 'patient276.txt', 'patient420.txt', 'patient959.txt', 'patient332.txt', 'patient193.txt', 'patient32.txt', 'patient311.txt', 'patient773.txt', 'patient845.txt', 'patient922.txt', 'patient982.txt', 'patient974.txt', 'patient889.txt', 'patient181.txt', 'patient437.txt', 'patient12.txt', 'patient285.txt', 'patient968.txt', 'patient703.txt', 'patient883.txt', 'patient981.txt']
    test_files = ['patient908.txt', 'patient212.txt', 'patient388.txt', 'patient933.txt', 'patient863.txt', 'patient59.txt', 'patient470.txt', 'patient487.txt', 'patient29.txt', 'patient717.txt', 'patient665.txt', 'patient909.txt', 'patient901.txt', 'patient154.txt', 'patient747.txt', 'patient222.txt', 'patient372.txt', 'patient310.txt', 'patient769.txt', 'patient206.txt', 'patient309.txt', 'patient152.txt', 'patient16.txt', 'patient789.txt', 'patient33.txt', 'patient236.txt', 'patient781.txt', 'patient687.txt', 'patient46.txt', 'patient738.txt', 'patient44.txt', 'patient23.txt', 'patient195.txt', 'patient42.txt', 'patient402.txt', 'patient82.txt', 'patient904.txt', 'patient572.txt', 'patient787.txt', 'patient944.txt', 'patient87.txt', 'patient899.txt', 'patient473.txt', 'patient423.txt', 'patient95.txt', 'patient689.txt', 'patient73.txt', 'patient172.txt', 'patient855.txt', 'patient123.txt', 'patient406.txt', 'patient43.txt', 'patient833.txt', 'patient246.txt', 'patient134.txt', 'patient956.txt', 'patient140.txt', 'patient109.txt', 'patient688.txt', 'patient830.txt', 'patient730.txt', 'patient56.txt', 'patient568.txt', 'patient427.txt', 'patient162.txt', 'patient15.txt', 'patient557.txt', 'patient567.txt', 'patient176.txt', 'patient321.txt', 'patient292.txt', 'patient771.txt', 'patient299.txt', 'patient168.txt', 'patient258.txt', 'patient18.txt', 'patient991.txt', 'patient637.txt', 'patient399.txt', 'patient893.txt', 'patient288.txt', 'patient144.txt', 'patient430.txt', 'patient135.txt', 'patient88.txt', 'patient262.txt', 'patient748.txt', 'patient507.txt', 'patient126.txt', 'patient45.txt', 'patient475.txt', 'patient105.txt', 'patient779.txt', 'patient729.txt', 'patient605.txt', 'patient927.txt', 'patient58.txt', 'patient343.txt', 'patient366.txt', 'patient692.txt']
    #data_dir = 'D:/srp/1000patients'
    data_dir = '/data/ziye/patient_notes'
    #train_files, validation_files, test_files = train_validation_test_split(data_dir)
    parser = argparse.ArgumentParser(description='Bug squash for Hierarchical Attention Networks')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1)
    parser.add_argument("--embed_dim", type=int, default=200)
    parser.add_argument("--label_file", type=str, default="result_csv_dead_los.csv")
    parser.add_argument("--vocab_file", type=str, default="vectors.txt")
    parser.add_argument("--train_files", type=list, default=train_files)
    parser.add_argument("--validation_files", type=list, default=validation_files)
    parser.add_argument("--test_files", type=list, default=test_files)
    parser.add_argument('--steps', type=int, default=10,help='perfom evaluation and model selection on validation default:100')
    parser.add_argument('--data_dir', type=str, default=data_dir)
    parser.add_argument('--output_dir', type=str, default=data_dir)
    # NOTE MODIFICATION (EMBEDDING)
    parser.add_argument("--pretrain", type=bool, default=True)
    parser.add_argument("--freeze", type=bool, default=True)

    # NOTE MODIFICATION (FEATURES)
    parser.add_argument("--dropout", type=float, default=0.1)

    config = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # NOTE MODIFICATION (FEATURE)
    if not os.path.exists(os.path.dirname('best_model_rl')):
        os.makedirs('best_model_rl', exist_ok=True)

    train(config, device)