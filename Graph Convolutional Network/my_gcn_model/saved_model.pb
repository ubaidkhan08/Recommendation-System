­
Е
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
П
MatrixDiagV3
diagonal"T
k
num_rows
num_cols
padding_value"T
output"T"	
Ttype"Q
alignstring
RIGHT_LEFT:2
0
LEFT_RIGHT
RIGHT_LEFT	LEFT_LEFTRIGHT_RIGHT

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
>
Minimum
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
Ѕ
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.11.02v2.11.0-rc2-15-g6290819256d8љђ
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
Д
.Adam/v/gcn_recommendation_model_6/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/v/gcn_recommendation_model_6/dense_6/bias
­
BAdam/v/gcn_recommendation_model_6/dense_6/bias/Read/ReadVariableOpReadVariableOp.Adam/v/gcn_recommendation_model_6/dense_6/bias*
_output_shapes
:*
dtype0
Д
.Adam/m/gcn_recommendation_model_6/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/m/gcn_recommendation_model_6/dense_6/bias
­
BAdam/m/gcn_recommendation_model_6/dense_6/bias/Read/ReadVariableOpReadVariableOp.Adam/m/gcn_recommendation_model_6/dense_6/bias*
_output_shapes
:*
dtype0
М
0Adam/v/gcn_recommendation_model_6/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *A
shared_name20Adam/v/gcn_recommendation_model_6/dense_6/kernel
Е
DAdam/v/gcn_recommendation_model_6/dense_6/kernel/Read/ReadVariableOpReadVariableOp0Adam/v/gcn_recommendation_model_6/dense_6/kernel*
_output_shapes

: *
dtype0
М
0Adam/m/gcn_recommendation_model_6/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *A
shared_name20Adam/m/gcn_recommendation_model_6/dense_6/kernel
Е
DAdam/m/gcn_recommendation_model_6/dense_6/kernel/Read/ReadVariableOpReadVariableOp0Adam/m/gcn_recommendation_model_6/dense_6/kernel*
_output_shapes

: *
dtype0
д
<Adam/v/gcn_recommendation_model_6/graph_convolution_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *M
shared_name><Adam/v/gcn_recommendation_model_6/graph_convolution_6/kernel
Э
PAdam/v/gcn_recommendation_model_6/graph_convolution_6/kernel/Read/ReadVariableOpReadVariableOp<Adam/v/gcn_recommendation_model_6/graph_convolution_6/kernel*
_output_shapes

:  *
dtype0
д
<Adam/m/gcn_recommendation_model_6/graph_convolution_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *M
shared_name><Adam/m/gcn_recommendation_model_6/graph_convolution_6/kernel
Э
PAdam/m/gcn_recommendation_model_6/graph_convolution_6/kernel/Read/ReadVariableOpReadVariableOp<Adam/m/gcn_recommendation_model_6/graph_convolution_6/kernel*
_output_shapes

:  *
dtype0
Я
9Adam/v/gcn_recommendation_model_6/embedding_13/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	и6*J
shared_name;9Adam/v/gcn_recommendation_model_6/embedding_13/embeddings
Ш
MAdam/v/gcn_recommendation_model_6/embedding_13/embeddings/Read/ReadVariableOpReadVariableOp9Adam/v/gcn_recommendation_model_6/embedding_13/embeddings*
_output_shapes
:	и6*
dtype0
Я
9Adam/m/gcn_recommendation_model_6/embedding_13/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	и6*J
shared_name;9Adam/m/gcn_recommendation_model_6/embedding_13/embeddings
Ш
MAdam/m/gcn_recommendation_model_6/embedding_13/embeddings/Read/ReadVariableOpReadVariableOp9Adam/m/gcn_recommendation_model_6/embedding_13/embeddings*
_output_shapes
:	и6*
dtype0
Я
9Adam/v/gcn_recommendation_model_6/embedding_12/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	'*J
shared_name;9Adam/v/gcn_recommendation_model_6/embedding_12/embeddings
Ш
MAdam/v/gcn_recommendation_model_6/embedding_12/embeddings/Read/ReadVariableOpReadVariableOp9Adam/v/gcn_recommendation_model_6/embedding_12/embeddings*
_output_shapes
:	'*
dtype0
Я
9Adam/m/gcn_recommendation_model_6/embedding_12/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	'*J
shared_name;9Adam/m/gcn_recommendation_model_6/embedding_12/embeddings
Ш
MAdam/m/gcn_recommendation_model_6/embedding_12/embeddings/Read/ReadVariableOpReadVariableOp9Adam/m/gcn_recommendation_model_6/embedding_12/embeddings*
_output_shapes
:	'*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
І
'gcn_recommendation_model_6/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'gcn_recommendation_model_6/dense_6/bias

;gcn_recommendation_model_6/dense_6/bias/Read/ReadVariableOpReadVariableOp'gcn_recommendation_model_6/dense_6/bias*
_output_shapes
:*
dtype0
Ў
)gcn_recommendation_model_6/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *:
shared_name+)gcn_recommendation_model_6/dense_6/kernel
Ї
=gcn_recommendation_model_6/dense_6/kernel/Read/ReadVariableOpReadVariableOp)gcn_recommendation_model_6/dense_6/kernel*
_output_shapes

: *
dtype0
Ц
5gcn_recommendation_model_6/graph_convolution_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *F
shared_name75gcn_recommendation_model_6/graph_convolution_6/kernel
П
Igcn_recommendation_model_6/graph_convolution_6/kernel/Read/ReadVariableOpReadVariableOp5gcn_recommendation_model_6/graph_convolution_6/kernel*
_output_shapes

:  *
dtype0
С
2gcn_recommendation_model_6/embedding_13/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	и6*C
shared_name42gcn_recommendation_model_6/embedding_13/embeddings
К
Fgcn_recommendation_model_6/embedding_13/embeddings/Read/ReadVariableOpReadVariableOp2gcn_recommendation_model_6/embedding_13/embeddings*
_output_shapes
:	и6*
dtype0
С
2gcn_recommendation_model_6/embedding_12/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	'*C
shared_name42gcn_recommendation_model_6/embedding_12/embeddings
К
Fgcn_recommendation_model_6/embedding_12/embeddings/Read/ReadVariableOpReadVariableOp2gcn_recommendation_model_6/embedding_12/embeddings*
_output_shapes
:	'*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
Џ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_12gcn_recommendation_model_6/embedding_12/embeddings2gcn_recommendation_model_6/embedding_13/embeddings5gcn_recommendation_model_6/graph_convolution_6/kernel)gcn_recommendation_model_6/dense_6/kernel'gcn_recommendation_model_6/dense_6/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_667132

NoOpNoOp
џ3
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*К3
valueА3B­3 BІ3
Ђ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
user_embedding
	item_embedding

	gcn_layer
dropout
output_layer
	optimizer

signatures*
'
0
1
2
3
4*
'
0
1
2
3
4*
* 
А
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
trace_3* 
6
trace_0
trace_1
trace_2
 trace_3* 
* 
 
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

embeddings*
 
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

embeddings*

-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

kernel*
Ѕ
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
9_random_generator* 
І
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

kernel
bias*

@
_variables
A_iterations
B_learning_rate
C_index_dict
D
_momentums
E_velocities
F_update_step_xla*

Gserving_default* 
rl
VARIABLE_VALUE2gcn_recommendation_model_6/embedding_12/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE2gcn_recommendation_model_6/embedding_13/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE5gcn_recommendation_model_6/graph_convolution_6/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)gcn_recommendation_model_6/dense_6/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'gcn_recommendation_model_6/dense_6/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
	1

2
3
4*

H0
I1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0*

0*
* 

Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

Otrace_0* 

Ptrace_0* 

0*

0*
* 

Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

Vtrace_0* 

Wtrace_0* 

0*

0*
* 

Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

]trace_0* 

^trace_0* 
* 
* 
* 

_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses* 

dtrace_0
etrace_1* 

ftrace_0
gtrace_1* 
* 

0
1*

0
1*
* 

hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

mtrace_0* 

ntrace_0* 
R
A0
o1
p2
q3
r4
s5
t6
u7
v8
w9
x10*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
o0
q1
s2
u3
w4*
'
p0
r1
t2
v3
x4*
* 
* 
8
y	variables
z	keras_api
	{total
	|count*
J
}	variables
~	keras_api
	total

count

_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
~
VARIABLE_VALUE9Adam/m/gcn_recommendation_model_6/embedding_12/embeddings1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE9Adam/v/gcn_recommendation_model_6/embedding_12/embeddings1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE9Adam/m/gcn_recommendation_model_6/embedding_13/embeddings1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE9Adam/v/gcn_recommendation_model_6/embedding_13/embeddings1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE<Adam/m/gcn_recommendation_model_6/graph_convolution_6/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE<Adam/v/gcn_recommendation_model_6/graph_convolution_6/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE0Adam/m/gcn_recommendation_model_6/dense_6/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE0Adam/v/gcn_recommendation_model_6/dense_6/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE.Adam/m/gcn_recommendation_model_6/dense_6/bias1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE.Adam/v/gcn_recommendation_model_6/dense_6/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*

{0
|1*

y	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

}	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
­
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameFgcn_recommendation_model_6/embedding_12/embeddings/Read/ReadVariableOpFgcn_recommendation_model_6/embedding_13/embeddings/Read/ReadVariableOpIgcn_recommendation_model_6/graph_convolution_6/kernel/Read/ReadVariableOp=gcn_recommendation_model_6/dense_6/kernel/Read/ReadVariableOp;gcn_recommendation_model_6/dense_6/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpMAdam/m/gcn_recommendation_model_6/embedding_12/embeddings/Read/ReadVariableOpMAdam/v/gcn_recommendation_model_6/embedding_12/embeddings/Read/ReadVariableOpMAdam/m/gcn_recommendation_model_6/embedding_13/embeddings/Read/ReadVariableOpMAdam/v/gcn_recommendation_model_6/embedding_13/embeddings/Read/ReadVariableOpPAdam/m/gcn_recommendation_model_6/graph_convolution_6/kernel/Read/ReadVariableOpPAdam/v/gcn_recommendation_model_6/graph_convolution_6/kernel/Read/ReadVariableOpDAdam/m/gcn_recommendation_model_6/dense_6/kernel/Read/ReadVariableOpDAdam/v/gcn_recommendation_model_6/dense_6/kernel/Read/ReadVariableOpBAdam/m/gcn_recommendation_model_6/dense_6/bias/Read/ReadVariableOpBAdam/v/gcn_recommendation_model_6/dense_6/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*"
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_667457
	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename2gcn_recommendation_model_6/embedding_12/embeddings2gcn_recommendation_model_6/embedding_13/embeddings5gcn_recommendation_model_6/graph_convolution_6/kernel)gcn_recommendation_model_6/dense_6/kernel'gcn_recommendation_model_6/dense_6/bias	iterationlearning_rate9Adam/m/gcn_recommendation_model_6/embedding_12/embeddings9Adam/v/gcn_recommendation_model_6/embedding_12/embeddings9Adam/m/gcn_recommendation_model_6/embedding_13/embeddings9Adam/v/gcn_recommendation_model_6/embedding_13/embeddings<Adam/m/gcn_recommendation_model_6/graph_convolution_6/kernel<Adam/v/gcn_recommendation_model_6/graph_convolution_6/kernel0Adam/m/gcn_recommendation_model_6/dense_6/kernel0Adam/v/gcn_recommendation_model_6/dense_6/kernel.Adam/m/gcn_recommendation_model_6/dense_6/bias.Adam/v/gcn_recommendation_model_6/dense_6/biastotal_1count_1totalcount*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_667530сь
С
Ш
O__inference_graph_convolution_6_layer_call_and_return_conditional_losses_667324

inputs

adj_matrix0
matmul_readvariableop_resource:  
identityЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ b
MatMul_1MatMul
adj_matrixMatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџ a
IdentityIdentityMatMul_1:product:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ ^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:џџџџџџџџџ :џџџџџџџџџџџџџџџџџџ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:\X
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
$
_user_specified_name
adj_matrix
и
c
E__inference_dropout_6_layer_call_and_return_conditional_losses_666839

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ЈQ
Ѕ
!__inference__wrapped_model_666761
input_1R
?gcn_recommendation_model_6_embedding_12_embedding_lookup_666719:	'R
?gcn_recommendation_model_6_embedding_13_embedding_lookup_666724:	и6_
Mgcn_recommendation_model_6_graph_convolution_6_matmul_readvariableop_resource:  S
Agcn_recommendation_model_6_dense_6_matmul_readvariableop_resource: P
Bgcn_recommendation_model_6_dense_6_biasadd_readvariableop_resource:
identityЂ9gcn_recommendation_model_6/dense_6/BiasAdd/ReadVariableOpЂ8gcn_recommendation_model_6/dense_6/MatMul/ReadVariableOpЂ8gcn_recommendation_model_6/embedding_12/embedding_lookupЂ8gcn_recommendation_model_6/embedding_13/embedding_lookupЂDgcn_recommendation_model_6/graph_convolution_6/MatMul/ReadVariableOp
.gcn_recommendation_model_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0gcn_recommendation_model_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
0gcn_recommendation_model_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
(gcn_recommendation_model_6/strided_sliceStridedSliceinput_17gcn_recommendation_model_6/strided_slice/stack:output:09gcn_recommendation_model_6/strided_slice/stack_1:output:09gcn_recommendation_model_6/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask
0gcn_recommendation_model_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
2gcn_recommendation_model_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
2gcn_recommendation_model_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      э
*gcn_recommendation_model_6/strided_slice_1StridedSliceinput_19gcn_recommendation_model_6/strided_slice_1/stack:output:0;gcn_recommendation_model_6/strided_slice_1/stack_1:output:0;gcn_recommendation_model_6/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskи
8gcn_recommendation_model_6/embedding_12/embedding_lookupResourceGather?gcn_recommendation_model_6_embedding_12_embedding_lookup_6667191gcn_recommendation_model_6/strided_slice:output:0*
Tindices0*R
_classH
FDloc:@gcn_recommendation_model_6/embedding_12/embedding_lookup/666719*'
_output_shapes
:џџџџџџџџџ*
dtype0
Agcn_recommendation_model_6/embedding_12/embedding_lookup/IdentityIdentityAgcn_recommendation_model_6/embedding_12/embedding_lookup:output:0*
T0*R
_classH
FDloc:@gcn_recommendation_model_6/embedding_12/embedding_lookup/666719*'
_output_shapes
:џџџџџџџџџЭ
Cgcn_recommendation_model_6/embedding_12/embedding_lookup/Identity_1IdentityJgcn_recommendation_model_6/embedding_12/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџк
8gcn_recommendation_model_6/embedding_13/embedding_lookupResourceGather?gcn_recommendation_model_6_embedding_13_embedding_lookup_6667243gcn_recommendation_model_6/strided_slice_1:output:0*
Tindices0*R
_classH
FDloc:@gcn_recommendation_model_6/embedding_13/embedding_lookup/666724*'
_output_shapes
:џџџџџџџџџ*
dtype0
Agcn_recommendation_model_6/embedding_13/embedding_lookup/IdentityIdentityAgcn_recommendation_model_6/embedding_13/embedding_lookup:output:0*
T0*R
_classH
FDloc:@gcn_recommendation_model_6/embedding_13/embedding_lookup/666724*'
_output_shapes
:џџџџџџџџџЭ
Cgcn_recommendation_model_6/embedding_13/embedding_lookup/Identity_1IdentityJgcn_recommendation_model_6/embedding_13/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџh
&gcn_recommendation_model_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Е
!gcn_recommendation_model_6/concatConcatV2Lgcn_recommendation_model_6/embedding_12/embedding_lookup/Identity_1:output:0Lgcn_recommendation_model_6/embedding_13/embedding_lookup/Identity_1:output:0/gcn_recommendation_model_6/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ z
 gcn_recommendation_model_6/ShapeShape*gcn_recommendation_model_6/concat:output:0*
T0*
_output_shapes
:z
0gcn_recommendation_model_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2gcn_recommendation_model_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2gcn_recommendation_model_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
*gcn_recommendation_model_6/strided_slice_2StridedSlice)gcn_recommendation_model_6/Shape:output:09gcn_recommendation_model_6/strided_slice_2/stack:output:0;gcn_recommendation_model_6/strided_slice_2/stack_1:output:0;gcn_recommendation_model_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskМ
&gcn_recommendation_model_6/eye/MinimumMinimum3gcn_recommendation_model_6/strided_slice_2:output:03gcn_recommendation_model_6/strided_slice_2:output:0*
T0*
_output_shapes
: g
$gcn_recommendation_model_6/eye/shapeConst*
_output_shapes
: *
dtype0*
valueB 
.gcn_recommendation_model_6/eye/concat/values_1Pack*gcn_recommendation_model_6/eye/Minimum:z:0*
N*
T0*
_output_shapes
:l
*gcn_recommendation_model_6/eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ќ
%gcn_recommendation_model_6/eye/concatConcatV2-gcn_recommendation_model_6/eye/shape:output:07gcn_recommendation_model_6/eye/concat/values_1:output:03gcn_recommendation_model_6/eye/concat/axis:output:0*
N*
T0*
_output_shapes
:n
)gcn_recommendation_model_6/eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Н
#gcn_recommendation_model_6/eye/onesFill.gcn_recommendation_model_6/eye/concat:output:02gcn_recommendation_model_6/eye/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџg
%gcn_recommendation_model_6/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : w
,gcn_recommendation_model_6/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџw
,gcn_recommendation_model_6/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџv
1gcn_recommendation_model_6/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    і
#gcn_recommendation_model_6/eye/diagMatrixDiagV3,gcn_recommendation_model_6/eye/ones:output:0.gcn_recommendation_model_6/eye/diag/k:output:05gcn_recommendation_model_6/eye/diag/num_rows:output:05gcn_recommendation_model_6/eye/diag/num_cols:output:0:gcn_recommendation_model_6/eye/diag/padding_value:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџв
Dgcn_recommendation_model_6/graph_convolution_6/MatMul/ReadVariableOpReadVariableOpMgcn_recommendation_model_6_graph_convolution_6_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0ы
5gcn_recommendation_model_6/graph_convolution_6/MatMulMatMul*gcn_recommendation_model_6/concat:output:0Lgcn_recommendation_model_6/graph_convolution_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ т
7gcn_recommendation_model_6/graph_convolution_6/MatMul_1MatMul,gcn_recommendation_model_6/eye/diag:output:0?gcn_recommendation_model_6/graph_convolution_6/MatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџ Ў
-gcn_recommendation_model_6/dropout_6/IdentityIdentityAgcn_recommendation_model_6/graph_convolution_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ К
8gcn_recommendation_model_6/dense_6/MatMul/ReadVariableOpReadVariableOpAgcn_recommendation_model_6_dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype0п
)gcn_recommendation_model_6/dense_6/MatMulMatMul6gcn_recommendation_model_6/dropout_6/Identity:output:0@gcn_recommendation_model_6/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџИ
9gcn_recommendation_model_6/dense_6/BiasAdd/ReadVariableOpReadVariableOpBgcn_recommendation_model_6_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0п
*gcn_recommendation_model_6/dense_6/BiasAddBiasAdd3gcn_recommendation_model_6/dense_6/MatMul:product:0Agcn_recommendation_model_6/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'gcn_recommendation_model_6/dense_6/ReluRelu3gcn_recommendation_model_6/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentity5gcn_recommendation_model_6/dense_6/Relu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџњ
NoOpNoOp:^gcn_recommendation_model_6/dense_6/BiasAdd/ReadVariableOp9^gcn_recommendation_model_6/dense_6/MatMul/ReadVariableOp9^gcn_recommendation_model_6/embedding_12/embedding_lookup9^gcn_recommendation_model_6/embedding_13/embedding_lookupE^gcn_recommendation_model_6/graph_convolution_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : : : 2v
9gcn_recommendation_model_6/dense_6/BiasAdd/ReadVariableOp9gcn_recommendation_model_6/dense_6/BiasAdd/ReadVariableOp2t
8gcn_recommendation_model_6/dense_6/MatMul/ReadVariableOp8gcn_recommendation_model_6/dense_6/MatMul/ReadVariableOp2t
8gcn_recommendation_model_6/embedding_12/embedding_lookup8gcn_recommendation_model_6/embedding_12/embedding_lookup2t
8gcn_recommendation_model_6/embedding_13/embedding_lookup8gcn_recommendation_model_6/embedding_13/embedding_lookup2
Dgcn_recommendation_model_6/graph_convolution_6/MatMul/ReadVariableOpDgcn_recommendation_model_6/graph_convolution_6/MatMul/ReadVariableOp:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Ѕ8
Ы
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_667215

inputs7
$embedding_12_embedding_lookup_667173:	'7
$embedding_13_embedding_lookup_667178:	и6D
2graph_convolution_6_matmul_readvariableop_resource:  8
&dense_6_matmul_readvariableop_resource: 5
'dense_6_biasadd_readvariableop_resource:
identityЂdense_6/BiasAdd/ReadVariableOpЂdense_6/MatMul/ReadVariableOpЂembedding_12/embedding_lookupЂembedding_13/embedding_lookupЂ)graph_convolution_6/MatMul/ReadVariableOpd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ј
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskь
embedding_12/embedding_lookupResourceGather$embedding_12_embedding_lookup_667173strided_slice:output:0*
Tindices0*7
_class-
+)loc:@embedding_12/embedding_lookup/667173*'
_output_shapes
:џџџџџџџџџ*
dtype0Х
&embedding_12/embedding_lookup/IdentityIdentity&embedding_12/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_12/embedding_lookup/667173*'
_output_shapes
:џџџџџџџџџ
(embedding_12/embedding_lookup/Identity_1Identity/embedding_12/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџю
embedding_13/embedding_lookupResourceGather$embedding_13_embedding_lookup_667178strided_slice_1:output:0*
Tindices0*7
_class-
+)loc:@embedding_13/embedding_lookup/667178*'
_output_shapes
:џџџџџџџџџ*
dtype0Х
&embedding_13/embedding_lookup/IdentityIdentity&embedding_13/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_13/embedding_lookup/667178*'
_output_shapes
:џџџџџџџџџ
(embedding_13/embedding_lookup/Identity_1Identity/embedding_13/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Щ
concatConcatV21embedding_12/embedding_lookup/Identity_1:output:01embedding_13/embedding_lookup/Identity_1:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ D
ShapeShapeconcat:output:0*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
eye/MinimumMinimumstrided_slice_2:output:0strided_slice_2:output:0*
T0*
_output_shapes
: L
	eye/shapeConst*
_output_shapes
: *
dtype0*
valueB Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 

eye/concatConcatV2eye/shape:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?l
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџL

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    д
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
)graph_convolution_6/MatMul/ReadVariableOpReadVariableOp2graph_convolution_6_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
graph_convolution_6/MatMulMatMulconcat:output:01graph_convolution_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
graph_convolution_6/MatMul_1MatMuleye/diag:output:0$graph_convolution_6/MatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџ x
dropout_6/IdentityIdentity&graph_convolution_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_6/MatMulMatMuldropout_6/Identity:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџi
IdentityIdentitydense_6/Relu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџѓ
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^embedding_12/embedding_lookup^embedding_13/embedding_lookup*^graph_convolution_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2>
embedding_12/embedding_lookupembedding_12/embedding_lookup2>
embedding_13/embedding_lookupembedding_13/embedding_lookup2V
)graph_convolution_6/MatMul/ReadVariableOp)graph_convolution_6/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Р4
і
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_667113
input_1&
embedding_12_667078:	'&
embedding_13_667081:	и6,
graph_convolution_6_667103:   
dense_6_667107: 
dense_6_667109:
identityЂdense_6/StatefulPartitionedCallЂ!dropout_6/StatefulPartitionedCallЂ$embedding_12/StatefulPartitionedCallЂ$embedding_13/StatefulPartitionedCallЂ+graph_convolution_6/StatefulPartitionedCalld
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      љ
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_1StridedSliceinput_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskљ
$embedding_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0embedding_12_667078*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_12_layer_call_and_return_conditional_losses_666785ћ
$embedding_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0embedding_13_667081*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_13_layer_call_and_return_conditional_losses_666798M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :С
concatConcatV2-embedding_12/StatefulPartitionedCall:output:0-embedding_13/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ D
ShapeShapeconcat:output:0*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
eye/MinimumMinimumstrided_slice_2:output:0strided_slice_2:output:0*
T0*
_output_shapes
: L
	eye/shapeConst*
_output_shapes
: *
dtype0*
valueB Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 

eye/concatConcatV2eye/shape:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?l
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџL

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    д
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
+graph_convolution_6/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0eye/diag:output:0graph_convolution_6_667103*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_graph_convolution_6_layer_call_and_return_conditional_losses_666830ј
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall4graph_convolution_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_666902
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0dense_6_667107dense_6_667109*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_666852w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^dense_6/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall%^embedding_12/StatefulPartitionedCall%^embedding_13/StatefulPartitionedCall,^graph_convolution_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall2L
$embedding_13/StatefulPartitionedCall$embedding_13/StatefulPartitionedCall2Z
+graph_convolution_6/StatefulPartitionedCall+graph_convolution_6/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1


-__inference_embedding_13_layer_call_fn_667298

inputs
unknown:	и6
identityЂStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_13_layer_call_and_return_conditional_losses_666798o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:џџџџџџџџџ: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Г
І
H__inference_embedding_12_layer_call_and_return_conditional_losses_667291

inputs*
embedding_lookup_667285:	'
identityЂembedding_lookupЕ
embedding_lookupResourceGatherembedding_lookup_667285inputs*
Tindices0**
_class 
loc:@embedding_lookup/667285*'
_output_shapes
:џџџџџџџџџ*
dtype0
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/667285*'
_output_shapes
:џџџџџџџџџ}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџs
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ё
c
*__inference_dropout_6_layer_call_fn_667334

inputs
identityЂStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_666902o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


є
C__inference_dense_6_layer_call_and_return_conditional_losses_666852

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Г
І
H__inference_embedding_13_layer_call_and_return_conditional_losses_666798

inputs*
embedding_lookup_666792:	и6
identityЂembedding_lookupЕ
embedding_lookupResourceGatherembedding_lookup_666792inputs*
Tindices0**
_class 
loc:@embedding_lookup/666792*'
_output_shapes
:џџџџџџџџџ*
dtype0
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/666792*'
_output_shapes
:џџџџџџџџџ}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџs
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


є
C__inference_dense_6_layer_call_and_return_conditional_losses_667371

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
я

;__inference_gcn_recommendation_model_6_layer_call_fn_666872
input_1
unknown:	'
	unknown_0:	и6
	unknown_1:  
	unknown_2: 
	unknown_3:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_666859o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Г
І
H__inference_embedding_13_layer_call_and_return_conditional_losses_667307

inputs*
embedding_lookup_667301:	и6
identityЂembedding_lookupЕ
embedding_lookupResourceGatherembedding_lookup_667301inputs*
Tindices0**
_class 
loc:@embedding_lookup/667301*'
_output_shapes
:џџџџџџџџџ*
dtype0
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/667301*'
_output_shapes
:џџџџџџџџџ}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџs
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ8
а
__inference__traced_save_667457
file_prefixQ
Msavev2_gcn_recommendation_model_6_embedding_12_embeddings_read_readvariableopQ
Msavev2_gcn_recommendation_model_6_embedding_13_embeddings_read_readvariableopT
Psavev2_gcn_recommendation_model_6_graph_convolution_6_kernel_read_readvariableopH
Dsavev2_gcn_recommendation_model_6_dense_6_kernel_read_readvariableopF
Bsavev2_gcn_recommendation_model_6_dense_6_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopX
Tsavev2_adam_m_gcn_recommendation_model_6_embedding_12_embeddings_read_readvariableopX
Tsavev2_adam_v_gcn_recommendation_model_6_embedding_12_embeddings_read_readvariableopX
Tsavev2_adam_m_gcn_recommendation_model_6_embedding_13_embeddings_read_readvariableopX
Tsavev2_adam_v_gcn_recommendation_model_6_embedding_13_embeddings_read_readvariableop[
Wsavev2_adam_m_gcn_recommendation_model_6_graph_convolution_6_kernel_read_readvariableop[
Wsavev2_adam_v_gcn_recommendation_model_6_graph_convolution_6_kernel_read_readvariableopO
Ksavev2_adam_m_gcn_recommendation_model_6_dense_6_kernel_read_readvariableopO
Ksavev2_adam_v_gcn_recommendation_model_6_dense_6_kernel_read_readvariableopM
Isavev2_adam_m_gcn_recommendation_model_6_dense_6_bias_read_readvariableopM
Isavev2_adam_v_gcn_recommendation_model_6_dense_6_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Й
valueЏBЌB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B §
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Msavev2_gcn_recommendation_model_6_embedding_12_embeddings_read_readvariableopMsavev2_gcn_recommendation_model_6_embedding_13_embeddings_read_readvariableopPsavev2_gcn_recommendation_model_6_graph_convolution_6_kernel_read_readvariableopDsavev2_gcn_recommendation_model_6_dense_6_kernel_read_readvariableopBsavev2_gcn_recommendation_model_6_dense_6_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableopTsavev2_adam_m_gcn_recommendation_model_6_embedding_12_embeddings_read_readvariableopTsavev2_adam_v_gcn_recommendation_model_6_embedding_12_embeddings_read_readvariableopTsavev2_adam_m_gcn_recommendation_model_6_embedding_13_embeddings_read_readvariableopTsavev2_adam_v_gcn_recommendation_model_6_embedding_13_embeddings_read_readvariableopWsavev2_adam_m_gcn_recommendation_model_6_graph_convolution_6_kernel_read_readvariableopWsavev2_adam_v_gcn_recommendation_model_6_graph_convolution_6_kernel_read_readvariableopKsavev2_adam_m_gcn_recommendation_model_6_dense_6_kernel_read_readvariableopKsavev2_adam_v_gcn_recommendation_model_6_dense_6_kernel_read_readvariableopIsavev2_adam_m_gcn_recommendation_model_6_dense_6_bias_read_readvariableopIsavev2_adam_v_gcn_recommendation_model_6_dense_6_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *$
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Е
_input_shapesЃ
 : :	':	и6:  : :: : :	':	':	и6:	и6:  :  : : ::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	':%!

_output_shapes
:	и6:$ 

_output_shapes

:  :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	':%	!

_output_shapes
:	':%
!

_output_shapes
:	и6:%!

_output_shapes
:	и6:$ 

_output_shapes

:  :$ 

_output_shapes

:  :$ 

_output_shapes

: :$ 

_output_shapes

: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
я

;__inference_gcn_recommendation_model_6_layer_call_fn_667021
input_1
unknown:	'
	unknown_0:	и6
	unknown_1:  
	unknown_2: 
	unknown_3:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_666993o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Г
І
H__inference_embedding_12_layer_call_and_return_conditional_losses_666785

inputs*
embedding_lookup_666779:	'
identityЂembedding_lookupЕ
embedding_lookupResourceGatherembedding_lookup_666779inputs*
Tindices0**
_class 
loc:@embedding_lookup/666779*'
_output_shapes
:џџџџџџџџџ*
dtype0
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/666779*'
_output_shapes
:џџџџџџџџџ}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџs
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Р

(__inference_dense_6_layer_call_fn_667360

inputs
unknown: 
	unknown_0:
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_666852o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

F
*__inference_dropout_6_layer_call_fn_667329

inputs
identityА
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_666839`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
М4
ѕ
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_666993

inputs&
embedding_12_666958:	'&
embedding_13_666961:	и6,
graph_convolution_6_666983:   
dense_6_666987: 
dense_6_666989:
identityЂdense_6/StatefulPartitionedCallЂ!dropout_6/StatefulPartitionedCallЂ$embedding_12/StatefulPartitionedCallЂ$embedding_13/StatefulPartitionedCallЂ+graph_convolution_6/StatefulPartitionedCalld
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ј
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskљ
$embedding_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0embedding_12_666958*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_12_layer_call_and_return_conditional_losses_666785ћ
$embedding_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0embedding_13_666961*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_13_layer_call_and_return_conditional_losses_666798M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :С
concatConcatV2-embedding_12/StatefulPartitionedCall:output:0-embedding_13/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ D
ShapeShapeconcat:output:0*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
eye/MinimumMinimumstrided_slice_2:output:0strided_slice_2:output:0*
T0*
_output_shapes
: L
	eye/shapeConst*
_output_shapes
: *
dtype0*
valueB Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 

eye/concatConcatV2eye/shape:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?l
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџL

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    д
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
+graph_convolution_6/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0eye/diag:output:0graph_convolution_6_666983*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_graph_convolution_6_layer_call_and_return_conditional_losses_666830ј
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall4graph_convolution_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_666902
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0dense_6_666987dense_6_666989*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_666852w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^dense_6/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall%^embedding_12/StatefulPartitionedCall%^embedding_13/StatefulPartitionedCall,^graph_convolution_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall2L
$embedding_13/StatefulPartitionedCall$embedding_13/StatefulPartitionedCall2Z
+graph_convolution_6/StatefulPartitionedCall+graph_convolution_6/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


-__inference_embedding_12_layer_call_fn_667282

inputs
unknown:	'
identityЂStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_12_layer_call_and_return_conditional_losses_666785o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:џџџџџџџџџ: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
С
Ш
O__inference_graph_convolution_6_layer_call_and_return_conditional_losses_666830

inputs

adj_matrix0
matmul_readvariableop_resource:  
identityЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ b
MatMul_1MatMul
adj_matrixMatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџ a
IdentityIdentityMatMul_1:product:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ ^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:џџџџџџџџџ :џџџџџџџџџџџџџџџџџџ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:\X
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
$
_user_specified_name
adj_matrix
ёa
Р
"__inference__traced_restore_667530
file_prefixV
Cassignvariableop_gcn_recommendation_model_6_embedding_12_embeddings:	'X
Eassignvariableop_1_gcn_recommendation_model_6_embedding_13_embeddings:	и6Z
Hassignvariableop_2_gcn_recommendation_model_6_graph_convolution_6_kernel:  N
<assignvariableop_3_gcn_recommendation_model_6_dense_6_kernel: H
:assignvariableop_4_gcn_recommendation_model_6_dense_6_bias:&
assignvariableop_5_iteration:	 *
 assignvariableop_6_learning_rate: _
Lassignvariableop_7_adam_m_gcn_recommendation_model_6_embedding_12_embeddings:	'_
Lassignvariableop_8_adam_v_gcn_recommendation_model_6_embedding_12_embeddings:	'_
Lassignvariableop_9_adam_m_gcn_recommendation_model_6_embedding_13_embeddings:	и6`
Massignvariableop_10_adam_v_gcn_recommendation_model_6_embedding_13_embeddings:	и6b
Passignvariableop_11_adam_m_gcn_recommendation_model_6_graph_convolution_6_kernel:  b
Passignvariableop_12_adam_v_gcn_recommendation_model_6_graph_convolution_6_kernel:  V
Dassignvariableop_13_adam_m_gcn_recommendation_model_6_dense_6_kernel: V
Dassignvariableop_14_adam_v_gcn_recommendation_model_6_dense_6_kernel: P
Bassignvariableop_15_adam_m_gcn_recommendation_model_6_dense_6_bias:P
Bassignvariableop_16_adam_v_gcn_recommendation_model_6_dense_6_bias:%
assignvariableop_17_total_1: %
assignvariableop_18_count_1: #
assignvariableop_19_total: #
assignvariableop_20_count: 
identity_22ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Й
valueЏBЌB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOpAssignVariableOpCassignvariableop_gcn_recommendation_model_6_embedding_12_embeddingsIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:м
AssignVariableOp_1AssignVariableOpEassignvariableop_1_gcn_recommendation_model_6_embedding_13_embeddingsIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_2AssignVariableOpHassignvariableop_2_gcn_recommendation_model_6_graph_convolution_6_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_3AssignVariableOp<assignvariableop_3_gcn_recommendation_model_6_dense_6_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_4AssignVariableOp:assignvariableop_4_gcn_recommendation_model_6_dense_6_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:Г
AssignVariableOp_5AssignVariableOpassignvariableop_5_iterationIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_6AssignVariableOp assignvariableop_6_learning_rateIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:у
AssignVariableOp_7AssignVariableOpLassignvariableop_7_adam_m_gcn_recommendation_model_6_embedding_12_embeddingsIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:у
AssignVariableOp_8AssignVariableOpLassignvariableop_8_adam_v_gcn_recommendation_model_6_embedding_12_embeddingsIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:у
AssignVariableOp_9AssignVariableOpLassignvariableop_9_adam_m_gcn_recommendation_model_6_embedding_13_embeddingsIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_10AssignVariableOpMassignvariableop_10_adam_v_gcn_recommendation_model_6_embedding_13_embeddingsIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:щ
AssignVariableOp_11AssignVariableOpPassignvariableop_11_adam_m_gcn_recommendation_model_6_graph_convolution_6_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:щ
AssignVariableOp_12AssignVariableOpPassignvariableop_12_adam_v_gcn_recommendation_model_6_graph_convolution_6_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_13AssignVariableOpDassignvariableop_13_adam_m_gcn_recommendation_model_6_dense_6_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_14AssignVariableOpDassignvariableop_14_adam_v_gcn_recommendation_model_6_dense_6_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_15AssignVariableOpBassignvariableop_15_adam_m_gcn_recommendation_model_6_dense_6_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_16AssignVariableOpBassignvariableop_16_adam_v_gcn_recommendation_model_6_dense_6_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_22IdentityIdentity_21:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_22Identity_22:output:0*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
и
c
E__inference_dropout_6_layer_call_and_return_conditional_losses_667339

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
@
Ы
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_667275

inputs7
$embedding_12_embedding_lookup_667226:	'7
$embedding_13_embedding_lookup_667231:	и6D
2graph_convolution_6_matmul_readvariableop_resource:  8
&dense_6_matmul_readvariableop_resource: 5
'dense_6_biasadd_readvariableop_resource:
identityЂdense_6/BiasAdd/ReadVariableOpЂdense_6/MatMul/ReadVariableOpЂembedding_12/embedding_lookupЂembedding_13/embedding_lookupЂ)graph_convolution_6/MatMul/ReadVariableOpd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ј
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskь
embedding_12/embedding_lookupResourceGather$embedding_12_embedding_lookup_667226strided_slice:output:0*
Tindices0*7
_class-
+)loc:@embedding_12/embedding_lookup/667226*'
_output_shapes
:џџџџџџџџџ*
dtype0Х
&embedding_12/embedding_lookup/IdentityIdentity&embedding_12/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_12/embedding_lookup/667226*'
_output_shapes
:џџџџџџџџџ
(embedding_12/embedding_lookup/Identity_1Identity/embedding_12/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџю
embedding_13/embedding_lookupResourceGather$embedding_13_embedding_lookup_667231strided_slice_1:output:0*
Tindices0*7
_class-
+)loc:@embedding_13/embedding_lookup/667231*'
_output_shapes
:џџџџџџџџџ*
dtype0Х
&embedding_13/embedding_lookup/IdentityIdentity&embedding_13/embedding_lookup:output:0*
T0*7
_class-
+)loc:@embedding_13/embedding_lookup/667231*'
_output_shapes
:џџџџџџџџџ
(embedding_13/embedding_lookup/Identity_1Identity/embedding_13/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Щ
concatConcatV21embedding_12/embedding_lookup/Identity_1:output:01embedding_13/embedding_lookup/Identity_1:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ D
ShapeShapeconcat:output:0*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
eye/MinimumMinimumstrided_slice_2:output:0strided_slice_2:output:0*
T0*
_output_shapes
: L
	eye/shapeConst*
_output_shapes
: *
dtype0*
valueB Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 

eye/concatConcatV2eye/shape:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?l
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџL

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    д
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
)graph_convolution_6/MatMul/ReadVariableOpReadVariableOp2graph_convolution_6_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
graph_convolution_6/MatMulMatMulconcat:output:01graph_convolution_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
graph_convolution_6/MatMul_1MatMuleye/diag:output:0$graph_convolution_6/MatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџ \
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_6/dropout/MulMul&graph_convolution_6/MatMul_1:product:0 dropout_6/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ m
dropout_6/dropout/ShapeShape&graph_convolution_6/MatMul_1:product:0*
T0*
_output_shapes
: 
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0e
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ф
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ ^
dropout_6/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Л
dropout_6/dropout/SelectV2SelectV2"dropout_6/dropout/GreaterEqual:z:0dropout_6/dropout/Mul:z:0"dropout_6/dropout/Const_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_6/MatMulMatMul#dropout_6/dropout/SelectV2:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџi
IdentityIdentitydense_6/Relu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџѓ
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^embedding_12/embedding_lookup^embedding_13/embedding_lookup*^graph_convolution_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2>
embedding_12/embedding_lookupembedding_12/embedding_lookup2>
embedding_13/embedding_lookupembedding_13/embedding_lookup2V
)graph_convolution_6/MatMul/ReadVariableOp)graph_convolution_6/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


d
E__inference_dropout_6_layer_call_and_return_conditional_losses_667351

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


d
E__inference_dropout_6_layer_call_and_return_conditional_losses_666902

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ч

4__inference_graph_convolution_6_layer_call_fn_667315

inputs

adj_matrix
unknown:  
identityЂStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputs
adj_matrixunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_graph_convolution_6_layer_call_and_return_conditional_losses_666830o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:џџџџџџџџџ :џџџџџџџџџџџџџџџџџџ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:\X
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
$
_user_specified_name
adj_matrix
3
б
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_666859

inputs&
embedding_12_666786:	'&
embedding_13_666799:	и6,
graph_convolution_6_666831:   
dense_6_666853: 
dense_6_666855:
identityЂdense_6/StatefulPartitionedCallЂ$embedding_12/StatefulPartitionedCallЂ$embedding_13/StatefulPartitionedCallЂ+graph_convolution_6/StatefulPartitionedCalld
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ј
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskљ
$embedding_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0embedding_12_666786*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_12_layer_call_and_return_conditional_losses_666785ћ
$embedding_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0embedding_13_666799*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_13_layer_call_and_return_conditional_losses_666798M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :С
concatConcatV2-embedding_12/StatefulPartitionedCall:output:0-embedding_13/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ D
ShapeShapeconcat:output:0*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
eye/MinimumMinimumstrided_slice_2:output:0strided_slice_2:output:0*
T0*
_output_shapes
: L
	eye/shapeConst*
_output_shapes
: *
dtype0*
valueB Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 

eye/concatConcatV2eye/shape:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?l
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџL

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    д
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
+graph_convolution_6/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0eye/diag:output:0graph_convolution_6_666831*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_graph_convolution_6_layer_call_and_return_conditional_losses_666830ш
dropout_6/PartitionedCallPartitionedCall4graph_convolution_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_666839
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0dense_6_666853dense_6_666855*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_666852w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџф
NoOpNoOp ^dense_6/StatefulPartitionedCall%^embedding_12/StatefulPartitionedCall%^embedding_13/StatefulPartitionedCall,^graph_convolution_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall2L
$embedding_13/StatefulPartitionedCall$embedding_13/StatefulPartitionedCall2Z
+graph_convolution_6/StatefulPartitionedCall+graph_convolution_6/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ
ы
$__inference_signature_wrapper_667132
input_1
unknown:	'
	unknown_0:	и6
	unknown_1:  
	unknown_2: 
	unknown_3:
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_666761o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
3
в
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_667067
input_1&
embedding_12_667032:	'&
embedding_13_667035:	и6,
graph_convolution_6_667057:   
dense_6_667061: 
dense_6_667063:
identityЂdense_6/StatefulPartitionedCallЂ$embedding_12/StatefulPartitionedCallЂ$embedding_13/StatefulPartitionedCallЂ+graph_convolution_6/StatefulPartitionedCalld
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      љ
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_1StridedSliceinput_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_maskљ
$embedding_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0embedding_12_667032*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_12_layer_call_and_return_conditional_losses_666785ћ
$embedding_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0embedding_13_667035*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_embedding_13_layer_call_and_return_conditional_losses_666798M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :С
concatConcatV2-embedding_12/StatefulPartitionedCall:output:0-embedding_13/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ D
ShapeShapeconcat:output:0*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
eye/MinimumMinimumstrided_slice_2:output:0strided_slice_2:output:0*
T0*
_output_shapes
: L
	eye/shapeConst*
_output_shapes
: *
dtype0*
valueB Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 

eye/concatConcatV2eye/shape:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?l
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџL

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    д
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
+graph_convolution_6/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0eye/diag:output:0graph_convolution_6_667057*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_graph_convolution_6_layer_call_and_return_conditional_losses_666830ш
dropout_6/PartitionedCallPartitionedCall4graph_convolution_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_666839
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0dense_6_667061dense_6_667063*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_666852w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџф
NoOpNoOp ^dense_6/StatefulPartitionedCall%^embedding_12/StatefulPartitionedCall%^embedding_13/StatefulPartitionedCall,^graph_convolution_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall2L
$embedding_13/StatefulPartitionedCall$embedding_13/StatefulPartitionedCall2Z
+graph_convolution_6/StatefulPartitionedCall+graph_convolution_6/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
ь

;__inference_gcn_recommendation_model_6_layer_call_fn_667147

inputs
unknown:	'
	unknown_0:	и6
	unknown_1:  
	unknown_2: 
	unknown_3:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_666859o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ь

;__inference_gcn_recommendation_model_6_layer_call_fn_667162

inputs
unknown:	'
	unknown_0:	и6
	unknown_1:  
	unknown_2: 
	unknown_3:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_666993o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs"
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ћ
serving_default
;
input_10
serving_default_input_1:0џџџџџџџџџ<
output_10
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:§Ё
З
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
user_embedding
	item_embedding

	gcn_layer
dropout
output_layer
	optimizer

signatures"
_tf_keras_model
C
0
1
2
3
4"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ё
trace_0
trace_1
trace_2
trace_32Ж
;__inference_gcn_recommendation_model_6_layer_call_fn_666872
;__inference_gcn_recommendation_model_6_layer_call_fn_667147
;__inference_gcn_recommendation_model_6_layer_call_fn_667162
;__inference_gcn_recommendation_model_6_layer_call_fn_667021П
ЖВВ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3

trace_0
trace_1
trace_2
 trace_32Ђ
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_667215
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_667275
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_667067
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_667113П
ЖВВ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2z trace_3
ЬBЩ
!__inference__wrapped_model_666761input_1"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Е
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
Е
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
Б
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

kernel"
_tf_keras_layer
М
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
9_random_generator"
_tf_keras_layer
Л
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer

@
_variables
A_iterations
B_learning_rate
C_index_dict
D
_momentums
E_velocities
F_update_step_xla"
experimentalOptimizer
,
Gserving_default"
signature_map
E:C	'22gcn_recommendation_model_6/embedding_12/embeddings
E:C	и622gcn_recommendation_model_6/embedding_13/embeddings
G:E  25gcn_recommendation_model_6/graph_convolution_6/kernel
;:9 2)gcn_recommendation_model_6/dense_6/kernel
5:32'gcn_recommendation_model_6/dense_6/bias
 "
trackable_list_wrapper
C
0
	1

2
3
4"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
;__inference_gcn_recommendation_model_6_layer_call_fn_666872input_1"П
ЖВВ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
B
;__inference_gcn_recommendation_model_6_layer_call_fn_667147inputs"П
ЖВВ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
B
;__inference_gcn_recommendation_model_6_layer_call_fn_667162inputs"П
ЖВВ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
B
;__inference_gcn_recommendation_model_6_layer_call_fn_667021input_1"П
ЖВВ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
ЇBЄ
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_667215inputs"П
ЖВВ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
ЇBЄ
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_667275inputs"П
ЖВВ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
ЈBЅ
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_667067input_1"П
ЖВВ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
ЈBЅ
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_667113input_1"П
ЖВВ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
ё
Otrace_02д
-__inference_embedding_12_layer_call_fn_667282Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zOtrace_0

Ptrace_02я
H__inference_embedding_12_layer_call_and_return_conditional_losses_667291Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zPtrace_0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
ё
Vtrace_02д
-__inference_embedding_13_layer_call_fn_667298Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zVtrace_0

Wtrace_02я
H__inference_embedding_13_layer_call_and_return_conditional_losses_667307Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zWtrace_0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object

]trace_02щ
4__inference_graph_convolution_6_layer_call_fn_667315А
ЇВЃ
FullArgSpec+
args# 
jself
jinputs
j
adj_matrix
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z]trace_0
Ё
^trace_02
O__inference_graph_convolution_6_layer_call_and_return_conditional_losses_667324А
ЇВЃ
FullArgSpec+
args# 
jself
jinputs
j
adj_matrix
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z^trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
Х
dtrace_0
etrace_12
*__inference_dropout_6_layer_call_fn_667329
*__inference_dropout_6_layer_call_fn_667334Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zdtrace_0zetrace_1
ћ
ftrace_0
gtrace_12Ф
E__inference_dropout_6_layer_call_and_return_conditional_losses_667339
E__inference_dropout_6_layer_call_and_return_conditional_losses_667351Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zftrace_0zgtrace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
ь
mtrace_02Я
(__inference_dense_6_layer_call_fn_667360Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zmtrace_0

ntrace_02ъ
C__inference_dense_6_layer_call_and_return_conditional_losses_667371Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zntrace_0
n
A0
o1
p2
q3
r4
s5
t6
u7
v8
w9
x10"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
C
o0
q1
s2
u3
w4"
trackable_list_wrapper
C
p0
r1
t2
v3
x4"
trackable_list_wrapper
П2МЙ
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
ЫBШ
$__inference_signature_wrapper_667132input_1"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
N
y	variables
z	keras_api
	{total
	|count"
_tf_keras_metric
`
}	variables
~	keras_api
	total

count

_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBо
-__inference_embedding_12_layer_call_fn_667282inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
H__inference_embedding_12_layer_call_and_return_conditional_losses_667291inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBо
-__inference_embedding_13_layer_call_fn_667298inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
H__inference_embedding_13_layer_call_and_return_conditional_losses_667307inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bџ
4__inference_graph_convolution_6_layer_call_fn_667315inputs
adj_matrix"А
ЇВЃ
FullArgSpec+
args# 
jself
jinputs
j
adj_matrix
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_graph_convolution_6_layer_call_and_return_conditional_losses_667324inputs
adj_matrix"А
ЇВЃ
FullArgSpec+
args# 
jself
jinputs
j
adj_matrix
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яBь
*__inference_dropout_6_layer_call_fn_667329inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
*__inference_dropout_6_layer_call_fn_667334inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_dropout_6_layer_call_and_return_conditional_losses_667339inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_dropout_6_layer_call_and_return_conditional_losses_667351inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
мBй
(__inference_dense_6_layer_call_fn_667360inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_dense_6_layer_call_and_return_conditional_losses_667371inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
J:H	'29Adam/m/gcn_recommendation_model_6/embedding_12/embeddings
J:H	'29Adam/v/gcn_recommendation_model_6/embedding_12/embeddings
J:H	и629Adam/m/gcn_recommendation_model_6/embedding_13/embeddings
J:H	и629Adam/v/gcn_recommendation_model_6/embedding_13/embeddings
L:J  2<Adam/m/gcn_recommendation_model_6/graph_convolution_6/kernel
L:J  2<Adam/v/gcn_recommendation_model_6/graph_convolution_6/kernel
@:> 20Adam/m/gcn_recommendation_model_6/dense_6/kernel
@:> 20Adam/v/gcn_recommendation_model_6/dense_6/kernel
::82.Adam/m/gcn_recommendation_model_6/dense_6/bias
::82.Adam/v/gcn_recommendation_model_6/dense_6/bias
.
{0
|1"
trackable_list_wrapper
-
y	variables"
_generic_user_object
:  (2total
:  (2count
/
0
1"
trackable_list_wrapper
-
}	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
!__inference__wrapped_model_666761n0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "3Њ0
.
output_1"
output_1џџџџџџџџџЊ
C__inference_dense_6_layer_call_and_return_conditional_losses_667371c/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
(__inference_dense_6_layer_call_fn_667360X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "!
unknownџџџџџџџџџЌ
E__inference_dropout_6_layer_call_and_return_conditional_losses_667339c3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ 
 Ќ
E__inference_dropout_6_layer_call_and_return_conditional_losses_667351c3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p
Њ ",Ђ)
"
tensor_0џџџџџџџџџ 
 
*__inference_dropout_6_layer_call_fn_667329X3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p 
Њ "!
unknownџџџџџџџџџ 
*__inference_dropout_6_layer_call_fn_667334X3Ђ0
)Ђ&
 
inputsџџџџџџџџџ 
p
Њ "!
unknownџџџџџџџџџ Њ
H__inference_embedding_12_layer_call_and_return_conditional_losses_667291^+Ђ(
!Ђ

inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
-__inference_embedding_12_layer_call_fn_667282S+Ђ(
!Ђ

inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџЊ
H__inference_embedding_13_layer_call_and_return_conditional_losses_667307^+Ђ(
!Ђ

inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
-__inference_embedding_13_layer_call_fn_667298S+Ђ(
!Ђ

inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџб
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_667067w@Ђ=
&Ђ#
!
input_1џџџџџџџџџ
Њ

trainingp ",Ђ)
"
tensor_0џџџџџџџџџ
 б
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_667113w@Ђ=
&Ђ#
!
input_1џџџџџџџџџ
Њ

trainingp",Ђ)
"
tensor_0џџџџџџџџџ
 а
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_667215v?Ђ<
%Ђ"
 
inputsџџџџџџџџџ
Њ

trainingp ",Ђ)
"
tensor_0џџџџџџџџџ
 а
V__inference_gcn_recommendation_model_6_layer_call_and_return_conditional_losses_667275v?Ђ<
%Ђ"
 
inputsџџџџџџџџџ
Њ

trainingp",Ђ)
"
tensor_0џџџџџџџџџ
 Ћ
;__inference_gcn_recommendation_model_6_layer_call_fn_666872l@Ђ=
&Ђ#
!
input_1џџџџџџџџџ
Њ

trainingp "!
unknownџџџџџџџџџЋ
;__inference_gcn_recommendation_model_6_layer_call_fn_667021l@Ђ=
&Ђ#
!
input_1џџџџџџџџџ
Њ

trainingp"!
unknownџџџџџџџџџЊ
;__inference_gcn_recommendation_model_6_layer_call_fn_667147k?Ђ<
%Ђ"
 
inputsџџџџџџџџџ
Њ

trainingp "!
unknownџџџџџџџџџЊ
;__inference_gcn_recommendation_model_6_layer_call_fn_667162k?Ђ<
%Ђ"
 
inputsџџџџџџџџџ
Њ

trainingp"!
unknownџџџџџџџџџх
O__inference_graph_convolution_6_layer_call_and_return_conditional_losses_667324^Ђ[
TЂQ
 
inputsџџџџџџџџџ 
-*

adj_matrixџџџџџџџџџџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ 
 П
4__inference_graph_convolution_6_layer_call_fn_667315^Ђ[
TЂQ
 
inputsџџџџџџџџџ 
-*

adj_matrixџџџџџџџџџџџџџџџџџџ
Њ "!
unknownџџџџџџџџџ Ё
$__inference_signature_wrapper_667132y;Ђ8
Ђ 
1Њ.
,
input_1!
input_1џџџџџџџџџ"3Њ0
.
output_1"
output_1џџџџџџџџџ