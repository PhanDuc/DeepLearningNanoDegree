#  Semi Supervised Learning

GAN for semi supervised learning: throw away the generator, but focus on Discrimination

- Previously: 
    - Discriminator output: FAKE v.s. REAL
- Now:
    - Discriminator output: Class 1(REAL), Class 2(REAL), .... and FAKE
    - $total_cost = coast_labeled + cost_unlabeled$
    - $cost_labeled = cross_entropy(logits, labels)$
    - $cost_unlabeled = cross_entropy(logits, real)$
    - $p_real = sum(softmax(real_classes))$

- feature matching
    - minimize the absolute differences in training data and generated samples

Details are in the notebook.