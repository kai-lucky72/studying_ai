import torch

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]
    ]
)



query = inputs[1]

attn_scores_2 = torch.empty(inputs.shape[0])  #attn_score calc
for i, x_i in enumerate(inputs):               #attn_score 1 way
    attn_scores_2[i] = torch.dot(x_i, query)
print("Attention score of 2nd input:",attn_scores_2)
res = 0                                           #attn_score 2 way
for idx,element in enumerate(inputs[0]):          #attn_score 2 way
    res += inputs[0][idx] * query[idx]




attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()     #weight calc 1 way
def softmax_naive(x):                                  #weight calc 2 way
    return torch.exp(x) / torch.exp(x).sum(dim=0)
attn_weights_2_naive = softmax_naive(attn_scores_2)   #weight calc 2 way
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)  #weight calc 3 way
print("Attention weights for 2nd input:", attn_weights_2)



query = inputs[1]  #2nd input token is the query
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2  += attn_weights_2[i] * x_i
print("the context vector of 2 input is:",context_vec_2)


attn_scores = torch.empty(6,6)                   #calculating attn_scores for all inputs 1 way
for i,x_i in enumerate(inputs):
    for j,x_j in enumerate(inputs):
        attn_scores[i, j] =torch.dot(x_i, x_j)
print("all attention scores for all inputs:",attn_scores)

attn_scores = inputs @ inputs.T                  #calc attn_scores forall inputs 2 way
print(attn_scores)



attn_weights = torch.softmax(attn_scores, dim=1)      # calc aattn_weights of all inputs
print("the weights of all inputs:", attn_weights)


print("All row sum to one:", attn_weights.sum(dim=1))


all_context_vecs =attn_weights @ inputs               #calc context vector of all inputs
print(all_context_vecs)




