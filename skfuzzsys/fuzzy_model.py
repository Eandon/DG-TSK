import torch
from torch import nn
from numpy.random import choice

from .membership_functions import *
from .t_norms import *
from .utils import *


class Antecedent(nn.Module):
    def __init__(self, in_dim, out_dim, num_fuzzy_set, mf='Gaussian', frb='CoCo-FRB', tnorm='prod',
                 fea_sel=False, gate_fea='gate1'):
        """
        :param in_dim: input dimension
        :param out_dim: output dimension
        :param num_fuzzy_set: No. of fuzzy sets defined on each feature
        :param mf: membership function, {'Gaussian' (default), 'simplified_Gaussian'}
        :param frb: fuzzy rule base {'CoCo-FRB' (default), 'FuCo-FRB'}
        :param tnorm: for computing firing strength, {'prod' (default), 'softmin', 'adasoftmin'}
        :param fea_sel: FS or not, {False (fault), True}
        :param gate_fea: gate function of FS, {'gate1' (default), 'gate2', 'gate3', 'gate_m'}
        """
        super(Antecedent, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_fuzzy_set = num_fuzzy_set
        self.mf = mf
        self.frb = frb
        self.tnorm = tnorm
        self.fea_sel = fea_sel

        if fea_sel:
            self.gate_fea_type = gate_fea
            self.gate_fea = eval(gate_fea)

        # CoCo-FRB or FuCo-FRB
        self.FRB = self._init_frb(in_dim, num_fuzzy_set, frb_type=frb)
        self.num_rule = self.FRB.size(0)

        # antecedent initialization
        if mf == 'Gaussian':
            partition = torch.arange(self.num_fuzzy_set, dtype=torch.float64) / (self.num_fuzzy_set - 1)
            self.center = nn.Parameter(partition.repeat([self.in_dim, 1]).T)  # [num_fuzzy_set, in_dim]
            self.spread = nn.Parameter(
                torch.ones([num_fuzzy_set, in_dim], dtype=torch.float64))  # [num_fuzzy_set, in_dim]
        elif mf == 'simplified_Gaussian':
            partition = torch.arange(self.num_fuzzy_set, dtype=torch.float64) / (self.num_fuzzy_set - 1)
            self.center = nn.Parameter(partition.repeat([self.in_dim, 1]).T)  # [num_fuzzy_set, in_dim]
        else:
            raise ValueError("Invalid value for mf: '{}'".format(mf))

        # feature gates initialization
        if fea_sel:
            if gate_fea == 'gate1':  # (- x ** 2).exp()
                self.gate_param_feature = nn.Parameter(torch.full([in_dim], -5.0))
            elif gate_fea == 'gate2':  # 1 - (- x ** 2).exp()
                self.gate_param_feature = nn.Parameter(torch.full([in_dim], 0.001))
            elif gate_fea == 'gate3':  # (- x ** 2).exp()
                self.gate_param_feature = nn.Parameter(torch.full([in_dim], 3.0))
            elif gate_fea == 'gate_m':  # x ** 2 * (1 - x ** 2).exp()
                self.gate_param_feature = nn.Parameter(torch.full([in_dim], 0.1))
            else:
                raise ValueError("Invalid value for gate_fea: '{}'".format(gate_fea))

    def _init_frb(self, in_dim, num_fuzzy_set, frb_type):
        """

        :param in_dim:  input dimension
        :param num_fuzzy_set: No. of fuzzy sets defined on each feature
        :param frb_type: fuzzy rule base, {'CoCo-FRB' (default), 'FuCo-FRB'}
        :return: the index of the fuzzy set for computing the firing strength
        """
        if frb_type == 'CoCo-FRB':
            fs_ind = torch.tensor(range(num_fuzzy_set)).unsqueeze(1).repeat_interleave(in_dim, dim=1)
            return fs_ind.long()
        elif frb_type == 'FuCo-FRB':
            fs_ind = torch.zeros([num_fuzzy_set ** in_dim, in_dim])
            for i, ii in enumerate(reversed(range(in_dim))):
                fs_ind[:, ii] = torch.tensor(range(num_fuzzy_set)).repeat_interleave(num_fuzzy_set ** i).repeat(
                    num_fuzzy_set ** ii)
            return fs_ind.long()
        else:
            raise ValueError(
                "Invalid value for frb: '{}', expected 'CoCo-FRB', 'FuCo-FRB'".format(self.frb))

    def reinit_gate_param(self, gate_param):
        self.gate_param_feature = nn.Parameter(torch.full(self.gate_param_feature.shape, gate_param))

    @property
    def gate_values(self):
        return self.gate_fea(self.gate_param_feature)

    def forward(self, model_input):
        """

        :param model_input:
        :return: firing strengths, [num_sam, num_rule]
        """
        model_input = model_input.double()  # [num_sam, in_dim]

        # membership values
        if self.mf == 'Gaussian':
            membership_value = gauss(model_input.unsqueeze(1), self.center, self.spread)  # [num_sam, num_rule, in_dim]
        elif self.mf == 'simplified_Gaussian':
            membership_value = sim_gauss(model_input.unsqueeze(1), self.center)  # [num_sam, num_rule, in_dim]
        else:
            raise ValueError("Invalid value for mf: '{}'".format(self.mf))

        # FS
        if self.fea_sel:
            membership_value = membership_value.pow(self.gate_fea(self.gate_param_feature))

        # firing strengths
        in_dim, fs_ind = self.in_dim, self.FRB
        if self.tnorm == 'prod':
            fir_str = membership_value[:, fs_ind, range(in_dim)].prod(dim=2)  # [num_sam, num_rule]
        elif self.tnorm == 'softmin':
            fir_str = softmin(membership_value[:, fs_ind, range(in_dim)], q=-12, dim=2)  # [num_sam, num_rule]
        elif self.tnorm == 'adasoftmin':
            fir_str = adasoftmin(membership_value[:, fs_ind, range(in_dim)], dim=2)  # [num_sam, num_rule]
        else:
            raise ValueError("Invalid value for tnorm: '{}'".format(self.tnorm))

        return fir_str  # [num_sam,num_rule]


class Consequent(nn.Module):
    def __init__(self, in_dim, out_dim, num_rule, order='first', rul_ext=False, gate_rule='gate1'):
        """
        :param in_dim: input dimension
        :param out_dim: output dimension
        :param num_rule: No. of fuzzy rules
        :param order: {'first' (default), 'zero'}
        :param rul_ext: RE or not, {False (fault), True}
        :param gate_rule: gate function for RE, {'gate1' (default), 'gate2', 'gate3', 'gate4', 'gate_m'}
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rule = num_rule
        self.order = order
        self.rul_ext = rul_ext

        # RE (rule extraction)
        if rul_ext:
            self.gate_rule_type = gate_rule
            self.gate_rule = eval(gate_rule)

        # consequent parameters initialization
        if order == 'first':
            self.con_param = nn.Parameter(torch.zeros([out_dim, num_rule, in_dim + 1],
                                                      dtype=torch.float64))  # [out_dim, num_rule, in_dim+1]
        elif self.order == 'zero':
            self.con_param = nn.Parameter(torch.zeros([out_dim, num_rule], dtype=torch.float64))  # [out_dim, num_rule]
        else:
            raise ValueError("Invalid value for order: '{}', expected 'first', 'zero'".format(self.order))

        # RE
        if rul_ext:
            if gate_rule == 'gate1':  # (- x ** 2).exp()
                self.gate_param_rule = nn.Parameter(torch.full([self.num_rule], -5.0))
            elif gate_rule == 'gate2':  # 1 - (- x ** 2).exp()
                self.gate_param_rule = nn.Parameter(torch.full([self.num_rule], 0.001))
            elif gate_rule == 'gate3':  # (- x ** 2).exp()
                self.gate_param_rule = nn.Parameter(torch.full([self.num_rule], 3.0))
            elif gate_rule == 'gate4':  # x * (1 - x ** 2).exp().sqrt()
                self.gate_param_rule = nn.Parameter(torch.full([self.num_rule], 0.01))
            elif gate_rule == 'gate_m':  # x ** 2 * (1 - x ** 2).exp()
                self.gate_param_rule = nn.Parameter(torch.full([self.num_rule], 0.1))
            else:
                raise ValueError("Invalid value for gate_rule: '{}'".format(gate_rule))

    def reinit_gate_param(self, gate_param):
        self.gate_param_rule = nn.Parameter(torch.full(self.gate_param_rule.shape, gate_param))

    def zero_to_first(self):
        if self.order == 'zero':
            self.order = 'first'
            self.con_param = nn.Parameter(torch.stack(
                [self.con_param.data] + [torch.zeros_like(self.con_param)] * self.in_dim, self.con_param.ndim))
        else:
            pass

    @property
    def gate_values(self):
        return self.gate_rule(self.gate_param_rule)

    def forward(self, model_input):
        """
        :param model_input:
        :return: rule output
        """
        if self.order == 'first':
            # [num_sam, num_rule, out_dim]
            rule_output = (self.con_param[:, :, 1:] @ model_input.T).T + self.con_param[:, :, 0].T
        elif self.order == 'zero':
            # [out_dim, num_rule]
            rule_output = self.con_param
        else:
            raise ValueError("Invalid value for order: '{}'".format(self.order))

        if self.rul_ext:
            if self.order == 'first':
                rule_output = rule_output * self.gate_rule(self.gate_param_rule).unsqueeze(1)
            elif self.order == 'zero':
                rule_output = rule_output * self.gate_rule(self.gate_param_rule)

        return rule_output


class TSK(nn.Module):
    def __init__(self, in_dim, out_dim, num_fuzzy_set, mf='Gaussian', frb='CoCo-FRB', tnorm='prod', order='first',
                 fea_sel=False, gate_fea='gate_m', rul_ext=False, gate_rule='gate_m'):
        """
        TSK fuzzy model
        :param in_dim: input dimension
        :param out_dim: output dimension
        :param num_fuzzy_set: No. of fuzzy sets defined on each feature
        :param mf: membership function, {'Gaussian' (default)}
        :param frb: fuzzy rule base, {'CoCo-FRB' (default), 'FuCo-FRB'}
        :param tnorm: T-norm for computing firing strength, {'prod' (default), 'softmin', 'adasoftmin'}
        :param order: {first (default), zero}
        :param fea_sel: FS or not, {False (default), True}
        :param gate_fea: gate function of FS, {'gate1' (default), 'gate_m'}
        :param rul_ext: RE or not, {False (default), True}
        :param gate_rule: gate function of RE, {'gate2' (default), 'gate_m'}
        """
        super(TSK, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_fuzzy_set = num_fuzzy_set
        self.mf = mf
        self.frb = frb
        self.tnorm = tnorm
        self.order = order

        # generate antecedent and consequent
        self.antecedent = Antecedent(in_dim, out_dim, num_fuzzy_set, mf, frb, tnorm, fea_sel, gate_fea)
        self.num_rule = self.antecedent.num_rule
        self.consequent = Consequent(in_dim, out_dim, self.num_rule, order, rul_ext, gate_rule)

    def reinit_system_points(self, model_input, target_output=None, spread=None, max_num_rule=None):
        """
        Point-based FRB (PFRB)
        initialize each point to a fuzzy rule
        :param model_input:
        :param target_output:
        :param spread:
        :param max_num_rule: maximum number of rules
        :return: the index of the samples selected for P-FRB
        """
        if max_num_rule is None:
            max_num_rule = model_input.size(0)

        # get which points are used for P-FRB
        if model_input.size(0) > max_num_rule and target_output is not None:
            # randomly select points according to class proportion
            rule_sam_ind_list = []
            sam_ind, label_scalar = target_output.nonzero(as_tuple=True)
            for each in label_scalar.unique():
                rule_sam_ind_list.append(torch.tensor(
                    choice(sam_ind[label_scalar == each],
                           round(max_num_rule / model_input.size(0) * label_scalar.eq(each).sum().tolist()),
                           replace=False)))
            rule_sam_ind_list = torch.cat(rule_sam_ind_list).tolist()
            rule_sam_ind_list.sort()
        elif model_input.size(0) > max_num_rule and target_output is None:
            # just randomly select points
            rule_sam_ind_list = choice(range(model_input.size(0)), max_num_rule, replace=False).tolist()
            rule_sam_ind_list.sort()
        else:  # all the points are selected for P-FRB
            rule_sam_ind_list = list(range(model_input.size(0)))

        # == antecedent
        rule_sam = model_input[rule_sam_ind_list]
        if self.antecedent.fea_sel:
            self.antecedent = Antecedent(self.in_dim, self.out_dim, rule_sam.size(0), self.mf, self.frb, self.tnorm,
                                         self.antecedent.fea_sel, self.antecedent.gate_fea_type)
        else:
            self.antecedent = Antecedent(self.in_dim, self.out_dim, rule_sam.size(0), self.mf, self.frb, self.tnorm)
        self.antecedent.center.data = nn.Parameter(rule_sam.double())  # [num_fuzzy_set, in_dim]

        # reinit the spread
        if type(spread) is float or type(spread) is int:
            self.antecedent.spread.data = torch.full(self.antecedent.spread.shape, spread, dtype=torch.float64)
        elif spread == 'std':
            self.antecedent.spread.data = model_input.std(0).repeat(self.antecedent.num_fuzzy_set, 1)
        elif spread == 'std_mean':
            self.antecedent.spread.data = rule_sam.std(0).mean().repeat(self.antecedent.num_fuzzy_set, self.in_dim)
        elif spread == 'std_sqrt':
            self.antecedent.spread.data = rule_sam.std(0).sqrt().repeat(self.antecedent.num_fuzzy_set, 1)
        elif spread == 'std_div2_sqrt':
            self.antecedent.spread.data = rule_sam.std(0).div(2).sqrt().repeat(self.antecedent.num_fuzzy_set, 1)
        else:
            raise ValueError(
                "Invalid value for spread: '{}', expected 'std', 'std_mean', 'std_sqrt', 'std_div2_sqrt'".format(
                    spread))

        # adjust the antecedent
        self.num_fuzzy_set = self.antecedent.num_fuzzy_set
        self.num_rule = self.antecedent.num_rule

        # == consequent
        if self.consequent.rul_ext:
            self.consequent = Consequent(self.in_dim, self.out_dim, self.num_rule, self.order,
                                         self.consequent.rul_ext, self.consequent.gate_rule_type)
        else:
            self.consequent = Consequent(self.in_dim, self.out_dim, self.num_rule, self.order)

        if target_output is not None:
            # reinit the consequent to the target outputs
            rule_sam_label = target_output[rule_sam_ind_list]
            if self.order == 'first':
                self.consequent.con_param.data[:, :, 0] = rule_sam_label.double().T  # [out_dim, num_rule, in_dim+1]
            elif self.order == 'zero':
                self.consequent.con_param.data = rule_sam_label.double().T  # [out_dim, num_rule]
            else:
                raise ValueError("Invalid value for tnorm: '{}'".format(self.tnorm))

        return rule_sam_ind_list

    def trained_param(self, tra_param='all'):
        """
        which parameters are going to be trained
        :param tra_param: {'IF', 'THEN', 'IF_THEN', 'gatesTHEN', 'all'(default)}
            IF: antecedent parameters; THEN: consequent parameters;
            IF_THEN: antecedent and consequent parameters;
            gatesTHEN: the gate parameters (both feature gates and rule gates) and consequent parameters
            all: all the parameters
        :return:
        """
        for each in self.parameters():
            each.requires_grad = False

        # which parameters needs gradients
        if tra_param == 'None':
            pass
        elif tra_param == 'IF':
            self.antecedent.center = nn.Parameter(self.antecedent.center)
            self.antecedent.spread = nn.Parameter(self.antecedent.spread)
        elif tra_param == 'THEN':
            self.consequent.con_param = nn.Parameter(self.consequent.con_param)
        elif tra_param == 'IF_THEN':
            self.antecedent.center = nn.Parameter(self.antecedent.center)
            self.antecedent.spread = nn.Parameter(self.antecedent.spread)
            self.consequent.con_param = nn.Parameter(self.consequent.con_param)
        elif tra_param == 'gatesTHEN':
            if self.antecedent.fea_sel:
                self.antecedent.gate_param_feature.requires_grad = True
            if self.consequent.rul_ext:
                self.consequent.gate_param_rule.requires_grad = True
            self.consequent.con_param.requires_grad = True
        elif tra_param == 'all':
            for each in self.parameters():
                each.requires_grad = True
        else:
            raise ValueError("Invalid value for tra_param: '{}'".format(tra_param))

    def zero_to_first(self):
        if self.order == 'zero':
            self.order = 'first'
            self.consequent.zero_to_first()
        else:
            pass

    def forward(self, model_input):
        """

        :param model_input: [num_sam, in_dim]
        :return: model outputs, [num_sam, out_dim]
        """
        model_input = model_input.double()

        # firing strengths * rule outputs
        fir_str = self.antecedent(model_input)
        rule_output = self.consequent(model_input)

        # de-fuzzy for computing the model outputs
        fir_str_bar = fir_str / fir_str.sum(dim=1).unsqueeze(1)  # [num_sam,num_rule]
        if self.order == 'first':
            model_output = torch.einsum('NRC,NR->NC', rule_output, fir_str_bar)  # [num_sam, out_dim]
        elif self.order == 'zero':
            model_output = fir_str_bar @ rule_output.T  # [num_sam, out_dim]
        else:
            raise ValueError("Invalid value for tnorm: '{}'".format(self.tnorm))

        return model_output

    def select_fea(self, tau_fea_para):
        """
        select features according to the gate values
        then tune the system parameters
        :param tau_fea_para: coefficient of the threshold of FS
        :return: the index of the selected features
        """
        # gate values * FS threshold * the index of the selected features
        gate_value_fea = self.antecedent.gate_fea(self.antecedent.gate_param_feature)
        tau_fea = threshold_fun(gate_value_fea.min(), gate_value_fea.max(), tau_fea_para)
        selected_fea_ind = gate_value_fea.gt(tau_fea).nonzero().squeeze(1).tolist()

        return selected_fea_ind

    def extract_rule(self, tau_rule_para):
        """
        extract rules according to the gate values
        then tune the system parameters
        :param tau_rule_para: coefficient of the threshold of RE
        :return: the index of the extracted rules
        """
        # gate values * RE threshold * the index of the extracted rules
        gate_value_rule = self.consequent.gate_rule(self.consequent.gate_param_rule)
        tau_rule = threshold_fun(gate_value_rule.min(), gate_value_rule.max(), tau_rule_para)
        extracted_rule_ind = gate_value_rule.abs().gt(tau_rule).nonzero().squeeze(1).tolist()

        # lower bound of the number of extracted rules, i.e., the number of the classes
        if len(extracted_rule_ind) < self.out_dim:
            extracted_rule_ind = gate_value_rule.sort(descending=True).indices[:self.out_dim].tolist()

        return extracted_rule_ind

    def prune_structure(self, selected_fea: list = None, extracted_rule: list = None):
        """
        prune the system according to the selected features or extracted rules
        :param selected_fea: the index of the selected features
        :param extracted_rule: the index of the extracted rules
        :return:
        """
        if selected_fea:
            # adjust the system parameters according to the selected features
            self.antecedent.center = nn.Parameter(self.antecedent.center.data[:, selected_fea])
            self.antecedent.spread = nn.Parameter(self.antecedent.spread.data[:, selected_fea])
            self.antecedent.FRB = self.antecedent.FRB[:, selected_fea]
            self.consequent.con_param = nn.Parameter(
                self.consequent.con_param.data[:, :, [0] + [i + 1 for i in selected_fea]])

            self.antecedent.in_dim = len(selected_fea)
            self.consequent.in_dim = self.antecedent.in_dim
            self.in_dim = self.antecedent.in_dim

        if extracted_rule:
            # adjust the system parameters according to the extracted rules
            self.antecedent.num_rule = len(extracted_rule)
            self.consequent.num_rule = self.antecedent.num_rule
            self.num_rule = len(extracted_rule)

            self.antecedent.FRB = self.antecedent.FRB[extracted_rule, :]
            self.consequent.con_param = nn.Parameter(self.consequent.con_param.data[:, extracted_rule, :])
