# -*- coding:utf-8 -*-

import json
import pickle as pkl

import numpy as np
from text2vec import SentenceModel

msd_dict = json.load(open('LLM_Ref/msd_dict.json', 'r', encoding='utf-8'))
disease_dict = json.load(open('LLM_Ref/disease_dict.json', 'r', encoding='utf-8'))


def get_ref(query: str):
    topic_list = query_range(query, k=1, bar=0.0)
    if len(topic_list) == 0:
        return None
    info_dict = info_list(topic_list)
    print(f"查询到info：\n{info_dict}\n查询到topics：\n{topic_list}\n")
    try:
        url = msd_dict[topic_list[0]]
    except:
        url = "www.msdmanuals.cn/professional"
    truncate_dict(info_dict)
    return info_dict, "https://" + url


def info_list(topic_list):
    info_base = {i: disease_dict[i] for i in topic_list}
    return info_base


def query_range(query: str, k: int = 3, bar=0.8):
    emb_d = pkl.load(open('LLM_Ref/MSD.pkl', 'rb'))
    embeddings = []
    for key, value in emb_d.items():
        embeddings.append(value)
    embeddings = np.asarray(embeddings)
    model = SentenceModel(model_name_or_path='LLM_Ref/text2vec')
    # model = SentenceModel()  # 从HF下载
    q_emb = model.encode(query)
    q_emb = q_emb / np.linalg.norm(q_emb, ord=2)

    # Calculate the cosine similarity between the query embedding and all other embeddings
    cos_similarities = np.dot(embeddings, q_emb)

    # Get the indices of the embeddings with the highest cosine similarity scores
    top_k_indices = cos_similarities.argsort()[-k:][::-1]
    print(cos_similarities[top_k_indices])
    sift_topK = top_k_indices[np.argwhere(cos_similarities[top_k_indices] > bar)]
    sift_topK = sift_topK.reshape(sift_topK.shape[0], )
    ret = []
    if len(sift_topK) == 0:
        return ret
    # for indices in top_k_indices:
    for indices in sift_topK:
        key = list(emb_d.keys())[indices]
        ret.append(key)
        print(key)
    return ret


def truncate_dict(d, max_length=200):
    def count_chars(d):
        total_chars = 0
        paths = []

        def recurse(sub_d, path):
            nonlocal total_chars
            if isinstance(sub_d, dict):
                for key, value in sub_d.items():
                    recurse(value, path + [key])
            elif isinstance(sub_d, str):
                total_chars += len(sub_d)
                paths.append((path, len(sub_d)))
            # 可以在这里添加对其他数据类型的处理

        recurse(d, [])
        return total_chars, paths

    def set_value(d, path, value):
        for key in path[:-1]:
            d = d[key]
        d[path[-1]] = value

    def truncate_chars(d, paths, total_chars, max_length):
        if total_chars <= max_length:
            return

        remaining_chars = max_length
        for path, length in sorted(paths, key=lambda x: x[1], reverse=True):
            if total_chars == 0:
                fraction = 0
            else:
                fraction = length / total_chars
            allowed_chars = int(fraction * max_length)
            current_value = get_value(d, path)
            if len(current_value) > allowed_chars:
                truncated = current_value[:max(0, allowed_chars - 3)] + '...' if allowed_chars > 3 else '...'
                set_value(d, path, truncated)
                remaining_chars -= len(truncated)
                total_chars -= length
            else:
                remaining_chars -= length

            if remaining_chars <= 0:
                break

    def get_value(d, path):
        for key in path:
            d = d[key]
        return d

    total_chars, paths = count_chars(d)
    truncate_chars(d, paths, total_chars, max_length)


if __name__ == '__main__':
    info_dict = {'巨细胞动脉炎': {
        '概述': '巨细胞动脉炎主要累及胸主动脉，颈部主动脉的大分支动脉以及颈动脉的颅外分支。常见合并风湿性多肌痛。症状和体征包括头痛、视力下降、颞动脉压痛、咀嚼时下颌肌肉疼痛。发热、体重减轻、身体不适、乏力等表现亦常见。ESR和C反应蛋白常常升高。诊断为临床诊断，并由颞动脉活检确诊。大剂量激素和/或托珠单抗和阿司匹林治疗通常有效且可以预防视力丧失。巨细胞动脉炎主要累及胸主动脉，颈部主动脉的大分支动脉以及颈动脉的颅外分支。常见合并风湿性多肌痛。症状和体征包括头痛、视力下降、颞动脉压痛、咀嚼时下颌肌肉疼痛。发热、体重减轻、身体不适、乏力等表现亦常见。ESR和C反应蛋白常常升高。诊断为临床诊断，并由颞动脉活检确诊。大剂量激素和/或托珠单抗和阿司匹林治疗通常有效且可以预防视力丧失。（也见血管炎概述）在美国和欧洲，巨细胞动脉炎比其他血管炎疾病更为常见。发病率根据种族不同而有所差异。尸检结果显示大量无症状患者。女性较多见。平均发病年龄为70岁左右，在50～>90岁之间。大约有40%～60%的巨细胞动脉炎患者合并风湿性多肌痛症状。颅内血管通常不受累。',
        '症状和体征': {
            '概述': ' 巨细胞动脉炎的症状可能在几周内逐渐出现或突然出现。 患者可能出现全身症状比如发热（通常低热）、乏力、身体不适、无法解释的体重减轻以及盗汗。某些患者初诊为不明原因的发热。最终，大部分患者产生了与受累动脉相关的症状。 严重时，有时跳动性头痛（颞动脉的、枕骨的、前额的或者弥漫的）是最常见的症状。可与触摸头皮或梳头时引起的头皮疼痛伴发。 视力损害包括复视、暗点、视力模糊、视野缺损（警示症状）。单眼短期内部分或全部视野缺损（一过性黑朦）之后可能很快出现永久性不可逆的视力丧失。如果不及时治疗，另外一只眼睛也可受累。尽管，完全的双侧失明很少见。视力减退是由于眼动脉分支动脉炎或后睫状体动脉炎，导致视神经缺血。眼底镜检查可见缺血性视神经炎的表现，如视神经乳头苍白和水肿、棉絮状渗出斑或者小的出血。后期，视神经萎缩。由于脑远端颈动脉区或基底部动脉病变导致的枕骨皮质梗死引起的中央区盲点，很少见。在过去50年中，视觉障碍的发病率有所下降，这可能是因为在视觉障碍发生之前，巨细胞动脉炎已经被识别和治疗。 间歇性运动障碍（缺血性肌肉疼痛）可发生在下颌肌肉和舌头或舌尖的肌肉。当咀嚼坚硬的食物时，下颌间歇性运动障碍尤其明显。间歇性下颌运动障碍与复视的高风险相关。 当出现颈动脉或脊柱基底动脉及其分支狭窄或阻塞时，可出现卒中、短暂性一过性缺血等神经系统表现。 胸主动脉瘤和主动脉夹层是严重的，通常是主动脉炎的晚期并发症，在没有其他症状的情况下可能会进展。 '},
        '诊断': {
            '概述': ' ESR 红细胞沉降率（ESR）、C反应蛋白和全血细胞计数（CBC） 动脉活检，通常用颞动脉 有时，颞动脉超声 在年龄>55岁的患者中，如果出现以下任何症状都应怀疑巨细胞动脉炎，尤其是他们有全身性炎症表现和实验室检查证据时： 新发的头痛 颈部以上动脉缺血而新发的症状或体征 咀嚼时出现下颌肌肉疼痛 颞动脉或头皮压痛 无法解释的慢性发热或贫血 如果患者伴有 风湿性多肌痛的症状 则诊断巨细胞动脉炎更明确。 体格检查可检测到颞动脉上的肿胀和压痛，有无结节或红斑，有时可触摸到脉搏消失。颞动脉可突起。颞动脉在检查者手指下呈现为饱满而不是塌陷状态，则提示异常。颈部大动脉和主动脉分支应评估是否受累。 如果怀疑该诊断时，可查ESR、CRP和CBC。大多数患者的ESR、C反应蛋白水平升高；慢性病性贫血亦常见。有时，血小板水平增高，而若检查血清白蛋白和总蛋白水平时，可发现蛋白水平低下。可见轻微的白细胞碎裂，但并不特异。 如果巨细胞动脉炎还不确诊，推荐活检一支动脉。因为炎症段与正常段经常交替，应该取材于看似异常的血管段。通常选择颞动脉症状部位进行活检，如果枕骨动脉看起来异常也可进行活检。颞动脉取材的最佳长度还不明确，但是长一些，约5cm长，可提高阳性率。对侧动脉活检的诊断价值很小。无需等到活检后才治疗。由于炎症消退缓慢，颞动脉活检可在皮质类固醇治疗开始后2周内完成。 当专家进行颞动脉彩色多普勒超声检查时，可检测到血管壁水肿，表现为光晕，并可替代颞动脉活检诊断巨细胞动脉炎。 1 ）。颞动脉超声应在治疗开始前或 5 天内进行，因为皮质类固醇会降低检测灵敏度。在众多优点中，该测试是无创的，不涉及辐射，并且可以对其他颅血管进行成像。然而，颞动脉超声的诊断有效性在很大程度上取决于超声操作者的技能和设备。 应在诊断时诊断后周期进行主动脉及其分支的影像学检查，即使在没有提示的症状或体征时也需随访（见表）。 诊断参考文献 1.Chrysidis S, Døhn UM, Terslev L, et al: Diagnostic accuracy of vascular ultrasound in patients with suspected giant cell arteritis (EUREKA): a prospective, multicentre, non-interventional, cohort study.The Lancet Rheumatology 3 (12) e865-e873, 2021.doi.org/10.1016/S2665-9913(21)00246-0 '},
        '治疗': {
            '概述': ' 糖皮质激素 低剂量阿司匹林 托珠单抗 从开始怀疑颞动脉炎开始，就应该治疗。即使活检报告延迟了2周，病理学特点仍是显而易见的。 经验与提示 如果>55岁的患者有新发头痛、间歇性下颌运动障碍、突发视力障碍和/或颞动脉压痛，应考虑立即用糖皮质激素治疗巨细胞动脉炎。 糖皮质激素是治疗的基础。糖皮质激素可快速缓解症状，防止大多数患者视力丧失。最佳的起始剂量、减量方式及总疗程还没有定论。泼尼松的开始剂量40～60mg，每天1次口服（或等效的激素）连续4周，然后逐渐减少，对于大多数患者来说都有效。 如果患者伴有视力损害，开始静脉使用甲泼尼龙500～1000mg/d，连续3～5天，可用以尝试防止视力进一步下降，特别是对侧眼睛。挽救视力可能更多地取决于皮质类固醇治疗的启动速度，而不是剂量。一旦发生视神经梗死，无论糖皮质激素的剂量多少，都不能恢复。 如果几周后症状消失，泼尼松可以逐渐减量，根据患者的反应，从约60毫克/天减少到40毫克/天，从每周5毫克到10毫克，从每周2.5毫克到5毫克到每天10毫克到20毫克，然后进一步减量，直到停药。不能单用ESR评价患者对治疗的反应（和疾病活动性）。例如，在老年患者中、单克隆丙种球蛋白病等其他因素也可升高ESR。应该用临床症状来评价。C-反应蛋白有时可能比ESR更为有用。 大多数患者至少需要激素治疗2年。长期服用激素有明显的不良反应，因此如果可行的话，应该减量。一半以上患者服用这些药物可出现药物相关的并发症。因此，正在研究替代的治疗方案。开始治疗时应考虑托珠单抗，一种IL-6 受体拮抗剂。托珠单抗是一种有效的选择，可以减少皮质类固醇激素的使用 (1 )。与皮质类固醇激素合用时，托珠单抗的疗效优于单独使用皮质类固醇激素（1-3 ).然而，托珠单抗治疗的持续时间尚未确定，由于存在憩室穿孔的风险，有憩室炎病史的患者应慎用该药。 对于长期服用泼尼松的老年患者应给予双膦酸盐化合物抗吸收药物以增加骨量，预防骨质疏松。 一项随机对照试验发现，抗肿瘤坏死因子药物英夫利昔单抗没有益处，且有潜在危害 (4 )。 低剂量阿司匹林（81-100mg/d，口服）可有助于防止缺血事件的发生，除非有限制，则应该给所有患者服用。 治疗参考文献 1.Adler S, Reichenbach S, Gloor A, et al: Risk of relapse after discontinuation of tocilizumab therapy in giant cell arteritis. Rheumatology (Oxford) 58(9):1639-1643, 2019.doi:10.1093/rheumatology/kez0913 2.Villiger PM, Adler S, Kuchen S, et al: Tocilizumab for induction and maintenance of remission in giant cell arteritis: A phase 2, randomised, double-blind, placebo-controlled trial.Lancet 387:1921–1927, 2016.doi: 10.1016/S0140-6736(16)00560-2 3.Stone JH, Tuckwell K, Dimonaco S, et al: Trial of tocilizumab in giant-cell arteritis.N Engl J Med 377:317–328, 2017.doi: 10.1056/NEJMoa1613849 4.Hoffman GS, Cid MC, Rendt-Zagar KE, et al: Infliximab for maintenance of glucocorticosteroid-induced remission of giant cell arteritis: a randomized trial.Ann Intern Med 146(9):621-30, 2007.doi: 10.7326/0003-4819-146-9-200705010-00004.PMID: 17470830. '}}}
    truncate_dict(info_dict)
    print(info_dict)
