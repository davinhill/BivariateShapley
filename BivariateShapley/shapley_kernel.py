import numpy as np
import pandas as pd
import scipy as sp
import logging
import copy
import itertools
import warnings
import shap
from shap.utils._legacy import convert_to_instance, convert_to_model, match_instance_to_data, match_model_to_data
from shap.utils._legacy import convert_to_instance_with_index, convert_to_link, IdentityLink, convert_to_data, DenseData, SparseData
from shap.utils import safe_isinstance
from scipy.special import binom
from scipy.sparse import issparse
from sklearn.linear_model import LassoLarsIC, Lasso, lars_path
from tqdm.auto import tqdm

log = logging.getLogger('shap')

class Bivariate_KernelExplainer(shap.KernelExplainer):

    def explain(self, incoming_instance, **kwargs):
        # convert incoming input to a standardized iml object
        instance = convert_to_instance(incoming_instance)
        #import pdb; pdb.set_trace()
        match_instance_to_data(instance, self.data)

        # find the feature groups we will test. If a feature does not change from its
        # current value then we know it doesn't impact the model
        self.varyingInds = self.varying_groups(instance.x)
        if self.data.groups is None:
            self.varyingFeatureGroups = np.array([i for i in self.varyingInds])
            self.M = self.varyingFeatureGroups.shape[0]
        else:
            self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
            self.M = len(self.varyingFeatureGroups)
            groups = self.data.groups
            # convert to numpy array as it is much faster if not jagged array (all groups of same length)
            if self.varyingFeatureGroups and all(len(groups[i]) == len(groups[0]) for i in self.varyingInds):
                self.varyingFeatureGroups = np.array(self.varyingFeatureGroups)
                # further performance optimization in case each group has a single value
                if self.varyingFeatureGroups.shape[1] == 1:
                    self.varyingFeatureGroups = self.varyingFeatureGroups.flatten()

        # find f(x)
        if self.keep_index:
            model_out = self.model.f(instance.convert_to_df())
        else:
            model_out = self.model.f(instance.x)
        if isinstance(model_out, (pd.DataFrame, pd.Series)):
            model_out = model_out.values
        self.fx = model_out[0]

        if not self.vector_out:
            self.fx = np.array([self.fx])

        # if no features vary then no feature has an effect
        if self.M == 0:
            phi = np.zeros((self.data.groups_size, self.D))
            phi_b = np.zeros((self.data.groups_size, self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))

        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros((self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))
            diff = self.link.f(self.fx) - self.link.f(self.fnull)
            phi_b = np.zeros((self.data.groups_size, self.data.groups_size, self.D))
            for d in range(self.D):
                phi[self.varyingInds[0],d] = diff[d]
                phi_b[self.varyingInds[0], :,d] = diff[d]/2
                phi_b[self.varyingInds[0], self.varyingInds[0], d] = 0



        # if more than one feature varies then we have to do real work
        else:
            self.l1_reg = kwargs.get("l1_reg", "auto")

            # pick a reasonable number of samples if the user didn't specify how many they wanted
            self.nsamples = kwargs.get("nsamples", "auto")
            if self.nsamples == "auto":
                self.nsamples = (2 * self.M + 2**11)*2

            # if we have enough samples to enumerate all subsets then ignore the unneeded samples
            self.max_samples = 2 ** 30
            if self.M <= 30:
                self.max_samples = 2 ** self.M - 2
                if self.nsamples > self.max_samples:
                    self.nsamples = self.max_samples

            # reserve space for some of our computations
            self.allocate()

            # weight the different subset sizes
            num_subset_sizes = np.int(np.ceil((self.M - 1) / 2.0))
            num_paired_subset_sizes = np.int(np.floor((self.M - 1) / 2.0))
            weight_vector = np.array([(self.M - 1.0) / (i * (self.M - i)) for i in range(1, num_subset_sizes + 1)])
            weight_vector[:num_paired_subset_sizes] *= 2
            weight_vector /= np.sum(weight_vector)
            log.debug("weight_vector = {0}".format(weight_vector))
            log.debug("num_subset_sizes = {0}".format(num_subset_sizes))
            log.debug("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
            log.debug("M = {0}".format(self.M))

            # fill out all the subset sizes we can completely enumerate
            # given nsamples*remaining_weight_vector[subset_size]
            num_full_subsets = 0
            num_samples_left = self.nsamples
            group_inds = np.arange(self.M, dtype='int64')
            mask = np.zeros(self.M)
            remaining_weight_vector = copy.copy(weight_vector)
            for subset_size in range(1, num_subset_sizes + 1):

                # determine how many subsets (and their complements) are of the current size
                nsubsets = binom(self.M, subset_size)
                if subset_size <= num_paired_subset_sizes: nsubsets *= 2
                log.debug("subset_size = {0}".format(subset_size))
                log.debug("nsubsets = {0}".format(nsubsets))
                log.debug("self.nsamples*weight_vector[subset_size-1] = {0}".format(
                    num_samples_left * remaining_weight_vector[subset_size - 1]))
                log.debug("self.nsamples*weight_vector[subset_size-1]/nsubsets = {0}".format(
                    num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets))

                # see if we have enough samples to enumerate all subsets of this size
                if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
                    num_full_subsets += 1
                    num_samples_left -= nsubsets

                    # rescale what's left of the remaining weight vector to sum to 1
                    if remaining_weight_vector[subset_size - 1] < 1.0:
                        remaining_weight_vector /= (1 - remaining_weight_vector[subset_size - 1])

                    # add all the samples of the current subset size
                    w = weight_vector[subset_size - 1] / binom(self.M, subset_size)
                    if subset_size <= num_paired_subset_sizes: w /= 2.0
                    for inds in itertools.combinations(group_inds, subset_size):
                        mask[:] = 0.0
                        mask[np.array(inds, dtype='int64')] = 1.0
                        self.addsample(instance.x, mask, w)
                        if subset_size <= num_paired_subset_sizes:
                            mask[:] = np.abs(mask - 1)
                            self.addsample(instance.x, mask, w)
                else:
                    break
            log.info("num_full_subsets = {0}".format(num_full_subsets))

            # add random samples from what is left of the subset space
            nfixed_samples = self.nsamplesAdded
            samples_left = self.nsamples - self.nsamplesAdded
            log.debug("samples_left = {0}".format(samples_left))
            if num_full_subsets != num_subset_sizes:
                remaining_weight_vector = copy.copy(weight_vector)
                remaining_weight_vector[:num_paired_subset_sizes] /= 2 # because we draw two samples each below
                remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
                remaining_weight_vector /= np.sum(remaining_weight_vector)
                log.info("remaining_weight_vector = {0}".format(remaining_weight_vector))
                log.info("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
                ind_set = np.random.choice(len(remaining_weight_vector), 4 * samples_left, p=remaining_weight_vector)
                ind_set_pos = 0
                used_masks = {}
                while samples_left > 0 and ind_set_pos < len(ind_set):
                    mask.fill(0.0)
                    ind = ind_set[ind_set_pos] # we call np.random.choice once to save time and then just read it here
                    ind_set_pos += 1
                    subset_size = ind + num_full_subsets + 1
                    mask[np.random.permutation(self.M)[:subset_size]] = 1.0

                    # only add the sample if we have not seen it before, otherwise just
                    # increment a previous sample's weight
                    mask_tuple = tuple(mask)
                    new_sample = False
                    if mask_tuple not in used_masks:
                        new_sample = True
                        used_masks[mask_tuple] = self.nsamplesAdded
                        samples_left -= 1
                        self.addsample(instance.x, mask, 1.0)
                    else:
                        self.kernelWeights[used_masks[mask_tuple]] += 1.0

                    # add the compliment sample
                    if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                        mask[:] = np.abs(mask - 1)

                        # only add the sample if we have not seen it before, otherwise just
                        # increment a previous sample's weight
                        if new_sample:
                            samples_left -= 1
                            self.addsample(instance.x, mask, 1.0)
                        else:
                            # we know the compliment sample is the next one after the original sample, so + 1
                            self.kernelWeights[used_masks[mask_tuple] + 1] += 1.0

                # normalize the kernel weights for the random samples to equal the weight left after
                # the fixed enumerated samples have been already counted
                weight_left = np.sum(weight_vector[num_full_subsets:])
                log.info("weight_left = {0}".format(weight_left))
                self.kernelWeights[nfixed_samples:] *= weight_left / self.kernelWeights[nfixed_samples:].sum()

            # execute the model on the synthetic samples we have created
            self.run()

            # solve then expand the feature importance (Shapley value) vector to contain the non-varying features
            phi = np.zeros((self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))

            #=================================================================
            # Code Modifications for Bivariate Shapley

            self.proj = None # the projection matrix (x^T x)^{-1} x^T. This matrix excludes the last feature in self.varyingInds
            self.proj_minus = None # the projection matrix (x^T x)^{-1} x^T. This matrix excludes the first feature in self.varyingInds

            # calculate univariate shapley
            for d in range(self.D):
                vphi, vphi_var = self.solve(self.nsamples / self.max_samples, d)
                phi[self.varyingInds, d] = vphi
                phi_var[self.varyingInds, d] = vphi_var

            self.phi_uni = phi


            # calculate bivariate shapley

            # initialize phi+ matrix; default value for phi_i(j) is phi_i/2 when j does not vary
            phi_b = np.repeat(np.expand_dims(phi, 1), self.data.groups_size, axis = 1).copy() / 2
            #phi_b = np.zeros((self.data.groups_size, self.data.groups_size, self.D))

            for idx, feat_j in enumerate(self.varyingInds):
                for d in range(self.D):
                    #vphi, _ = self.solve(self.nsamples / self.max_samples, d, self.maskMatrix[:,idx] == 0)
                    vphi, _ = self.solve(self.nsamples / self.max_samples, d, self.maskMatrix[:,idx] == 1)
                    phi_b[:,feat_j,d] = 0
                    phi_b[self.varyingInds, feat_j, d] = vphi
                    phi_b[feat_j,feat_j,d] = 0
            #=================================================================

        if not self.vector_out:
            phi = np.squeeze(phi, axis=1)
            phi_var = np.squeeze(phi_var, axis=1)
            phi_b = np.squeeze(phi_b, axis = 2)
        
        #np.fill_diagonal(phi_b, 0)
        self.phi_b = phi_b.transpose()
        return phi


    def solve(self, fraction_evaluated, dim, j_present_ind = None):

        #=================================================================
        # Code Modifications for Bivariate Shapley

        if j_present_ind is not None:
            # phi+
            tmp = self.ey[:,dim] * (j_present_ind)  # labels where j not in S are set to zero
            eyAdj = self.linkfv(tmp) - self.link.f(0) # new adjusted labels
            fnull = 0

            # phi-
            tmp = self.ey[:,dim] * (1 - j_present_ind) # labels where j in S are set to zero
            eyAdj_minus = self.linkfv(tmp) - self.link.f(self.fnull[dim]) # new adjusted labels

        else:
            eyAdj = self.linkfv(self.ey[:,dim]) - self.link.f(self.fnull[dim]) 
            fnull = self.fnull[dim]

        #=================================================================
        s = np.sum(self.maskMatrix, 1)

        # do feature selection if we have not well enumerated the space
        nonzero_inds = np.arange(self.M)
        log.debug("fraction_evaluated = {0}".format(fraction_evaluated))
        # if self.l1_reg == "auto":
        #     warnings.warn(
        #         "l1_reg=\"auto\" is deprecated and in the next version (v0.29) the behavior will change from a " \
        #         "conditional use of AIC to simply \"num_features(10)\"!"
        #     )
        if (self.l1_reg not in ["auto", False, 0]) or (fraction_evaluated < 0.2 and self.l1_reg == "auto"):
            w_aug = np.hstack((self.kernelWeights * (self.M - s), self.kernelWeights * s))
            log.info("np.sum(w_aug) = {0}".format(np.sum(w_aug)))
            log.info("np.sum(self.kernelWeights) = {0}".format(np.sum(self.kernelWeights)))
            w_sqrt_aug = np.sqrt(w_aug)
            eyAdj_aug = np.hstack((eyAdj, eyAdj - (self.link.f(self.fx[dim]) - self.link.f(fnull))))
            eyAdj_aug *= w_sqrt_aug
            mask_aug = np.transpose(w_sqrt_aug * np.transpose(np.vstack((self.maskMatrix, self.maskMatrix - 1))))
            #var_norms = np.array([np.linalg.norm(mask_aug[:, i]) for i in range(mask_aug.shape[1])])

            # select a fixed number of top features
            if isinstance(self.l1_reg, str) and self.l1_reg.startswith("num_features("):
                r = int(self.l1_reg[len("num_features("):-1])
                nonzero_inds = lars_path(mask_aug, eyAdj_aug, max_iter=r)[1]

            # use an adaptive regularization method
            elif self.l1_reg == "auto" or self.l1_reg == "bic" or self.l1_reg == "aic":
                c = "aic" if self.l1_reg == "auto" else self.l1_reg
                nonzero_inds = np.nonzero(LassoLarsIC(criterion=c).fit(mask_aug, eyAdj_aug).coef_)[0]

            # use a fixed regularization coeffcient
            else:
                nonzero_inds = np.nonzero(Lasso(alpha=self.l1_reg).fit(mask_aug, eyAdj_aug).coef_)[0]

        if len(nonzero_inds) == 0:
            return np.zeros(self.M), np.ones(self.M)

        # eliminate one variable with the constraint that all features sum to the output
        eyAdj2 = eyAdj - self.maskMatrix[:, nonzero_inds[-1]] * (self.link.f(self.fx[dim]) - self.link.f(fnull))

        if self.proj is None:  # if the projection matrix has not previously been calculated
            etmp = np.transpose(np.transpose(self.maskMatrix[:, nonzero_inds[:-1]]) - self.maskMatrix[:, nonzero_inds[-1]])
            log.debug("etmp[:4,:] {0}".format(etmp[:4, :]))

            # solve a weighted least squares equation to estimate phi
            tmp = np.transpose(np.transpose(etmp) * np.transpose(self.kernelWeights))
            etmp_dot = np.dot(np.transpose(tmp), etmp)
            try:
                tmp2 = np.linalg.inv(etmp_dot)
            except np.linalg.LinAlgError:
                tmp2 = np.linalg.pinv(etmp_dot)
                warnings.warn(
                    "Linear regression equation is singular, Moore-Penrose pseudoinverse is used instead of the regular inverse.\n"
                    "To use regular inverse do one of the following:\n"
                    "1) turn up the number of samples,\n"
                    "2) turn up the L1 regularization with num_features(N) where N is less than the number of samples,\n"
                    "3) group features together to reduce the number of inputs that need to be explained."
                )
            self.proj = np.dot(tmp2, np.transpose(tmp))
        w = np.dot(self.proj, eyAdj2)

        ############################
        # Calculate Phi-
        if j_present_ind is not None:
            eyAdj2 = eyAdj_minus - self.maskMatrix[:, nonzero_inds[0]] * (self.link.f(0) - self.link.f(self.fnull[dim]))
            if self.proj_minus is None: # if the projection matrix has not previously been calculated

                # Note that the deleted indices are different
                etmp = np.transpose(np.transpose(self.maskMatrix[:, nonzero_inds[1:]]) - self.maskMatrix[:, nonzero_inds[0]])

                # solve a weighted least squares equation to estimate phi
                tmp = np.transpose(np.transpose(etmp) * np.transpose(self.kernelWeights))
                etmp_dot = np.dot(np.transpose(tmp), etmp)
                try:
                    tmp2 = np.linalg.inv(etmp_dot)
                except np.linalg.LinAlgError:
                    tmp2 = np.linalg.pinv(etmp_dot)
                    warnings.warn(
                        "Linear regression equation is singular, Moore-Penrose pseudoinverse is used instead of the regular inverse.\n"
                        "To use regular inverse do one of the following:\n"
                        "1) turn up the number of samples,\n"
                        "2) turn up the L1 regularization with num_features(N) where N is less than the number of samples,\n"
                        "3) group features together to reduce the number of inputs that need to be explained."
                    )
                self.proj_minus = np.dot(tmp2, np.transpose(tmp))
            w_minus = np.dot(self.proj_minus, eyAdj2)

            # calculate phi+ using the property that phi+ = phi - phi-

        ######

        log.debug("np.sum(w) = {0}".format(np.sum(w)))
        log.debug("self.link(self.fx) - self.link(fnull) = {0}".format(
            self.link.f(self.fx[dim]) - self.link.f(fnull)))
        log.debug("self.fx = {0}".format(self.fx[dim]))
        log.debug("self.link(self.fx) = {0}".format(self.link.f(self.fx[dim])))
        log.debug("fnull = {0}".format(fnull))
        log.debug("self.link(fnull) = {0}".format(self.link.f(fnull)))

        phi = np.zeros(self.M)
        phi[nonzero_inds[:-1]] = w # assign shapley values

        # calculate shapley value for deleted feature
        if j_present_ind is None:
            # univariate
            phi[nonzero_inds[-1]] = (self.link.f(self.fx[dim]) - self.link.f(fnull)) - sum(w)
        else:
            # bivariate, use the property that phi+ = phi - phi-
            phi[nonzero_inds[-1]] = self.phi_uni[self.varyingInds[-1]] - w_minus[-1]

        log.info("phi = {0}".format(phi))

        # clean up any rounding errors
        for i in range(self.M):
            if np.abs(phi[i]) < 1e-10:
                phi[i] = 0

        return phi, np.ones(len(phi))
    
