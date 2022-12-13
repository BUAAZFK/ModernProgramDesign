from GraphStat.NetworkBuilder.graph import *
from GraphStat.NetworkBuilder.node import *
from GraphStat.NetworkBuilder.stat import *
from GraphStat.Visualization.plotgraph import *
from GraphStat.Visualization.plotnodes import *
import pickle as pl
print('程序开始运行！')


# 测试graph的文件
graph = init_graph()
print('图已初始化并加载信息！')

# with open('./graph', 'wb') as f:
#     pl.dump(f, './graph')

# G = pl.load('./graph')
# print(graph)
# save_graph(graph)
# print('图已序列化并保存至同级目录！')


# graph_ = load_graph('./test.gpickle')
# print('反序列化完成！')
# print('graph:\n', graph)


# # 测试node的文件
# nodes = init_node()
# print('已生成nodes字典！')
# keys = list(nodes.keys())
# for i in range(10):
#     print(nodes[keys[i]])

id = int(input('请输入想要获取信息的节点的ID:'))
# views_id = get('views', id)
# print(f'已获取id为{id}的views信息！')
# print('id', views_id)

# print_node(id)
# print(f'已打印{id}的全部信息！')


# # 测试static的文件
# node_nums = get_node_number(graph)
# print('已获取图中点的总数！')
# print('node_nums:', node_nums)


# edge_nums = get_edge_number(graph)
# print('已获取图中边的总数！')
# print('edge_nums:', edge_nums)


# average_degree = cal_average_degree(graph)
# print('已获取图的平均度！')
# print('average_degree:', average_degree)


# degree = cal_degree_distribution(graph)
# print('已获取图中度的分布！')
# print('degree:\n', (deg for deg in degree[:10]), '……')


# views = cal_view_distribution(graph)
# print('已获取图中views的分布！')
# print('views:\n', (view for view in views[:10]), '……')


# # 测试plotgraph的文件
#plot_ego(graph, id)
#print(f'已绘制id为{id}的ego图并保存至当前路径！')
# plotdegree_distribution(graph)
# print(f'已绘制度的分布图！并保存至当前路径')

# # 测试plotnodes的文件
attr = input('请输入需要绘制分布图的信息名称(views,mature,life_time,created_at,updated_at,numeric_id,dead_account,language,affiliate): ')
plot_nodes_attr(graph, attr)
# print(f'已绘制{attr}属性的分布图并保存至当前目录！')