#include <memory>
#include <torch/script.h>
#include <torch/custom_class.h>

enum OcType
{
    NONLEAF = -1,
    SURFACE = 0,
    FEATURE = 1
};
// 八叉树中每一个节点的定义
class Octant : public torch::CustomClassHolder
{
public:
    // 会建议编译器将函数体直接插入到每个调用点 可以减少函数调用的开销 
    // 提高程序的执行效率
    inline Octant()
    {
        code_ = 0;
        side_ = 0;
        // index_ = next_index_++;
        depth_ = -1;
        is_leaf_ = false;
        children_mask_ = 0;
        type_ = NONLEAF;
        for (unsigned int i = 0; i < 8; i++)
        {
            child_ptr_[i] = nullptr;
            // feature_index_[i] = -1;
        }
    }
    ~Octant() {}

    // std::shared_ptr<Octant> &child(const int x, const int y, const int z)
    // {
    //     return child_ptr_[x + y * 2 + z * 4];
    // };

    // std::shared_ptr<Octant> &child(const int offset)
    // {
    //     return child_ptr_[offset];
    // }
    Octant *&child(const int x, const int y, const int z)
    {
        return child_ptr_[x + y * 2 + z * 4];
    };

    Octant *&child(const int offset)
    {
        return child_ptr_[offset];
    }

    uint64_t code_;
    bool is_leaf_;
    unsigned int side_;
    unsigned char children_mask_;
    // std::shared_ptr<Octant> child_ptr_[8];
    // int feature_index_[8];
    int index_;
    int depth_;
    int type_;
    // int feat_index_;
    Octant *child_ptr_[8];
    static int next_index_;
};
// 八叉树的定义
class Octree : public torch::CustomClassHolder
{
public:
    // 两个构造函数  一个有参 一个无参
    Octree();
    // temporal solution
    Octree(int64_t grid_dim, int64_t feat_dim, double voxel_size, std::vector<torch::Tensor> all_pts);
    // 析构函数 是没有参数的  在一个对象即将被销毁时调用 释放对象可能拥有的资源
    ~Octree();
    void init(int64_t grid_dim, int64_t feat_dim, double voxel_size);

    // allocate voxels
    void insert(torch::Tensor vox);
    double try_insert(torch::Tensor pts);
    // 第二棵树的插入函数
    void insert2(torch::Tensor vox);
    // 第三棵树的插入函数
    void insert3(torch::Tensor vox);

    // find a particular octant
    Octant *find_octant(std::vector<float> coord);
    Octant *find_octant_fa(std::vector<float> coord);
    Octant *find_octant_fafa(std::vector<float> coord);

    // test intersections
    bool has_voxel(torch::Tensor pose);

    // query features
    torch::Tensor get_features(torch::Tensor pts);

    // get all voxels
    torch::Tensor get_voxels();
    std::vector<float> get_voxel_recursive(Octant *n);

    // get leaf voxels
    torch::Tensor get_leaf_voxels();
    std::vector<float> get_leaf_voxel_recursive(Octant *n);

    // count nodes
    int64_t count_nodes();
    int64_t count_recursive(Octant *n);

    // count leaf nodes
    int64_t count_leaf_nodes();
    // int64_t leaves_count_recursive(std::shared_ptr<Octant> n);
    int64_t leaves_count_recursive(Octant *n);

    // get voxel centres and childrens
    // std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_centres_and_children();
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get_centres_and_children();

public:
    // 八叉树的最大节点数
    int size_;
    int size2_;
    int size3_;
    // embedding维度
    int feat_dim_;
    // 最大层数
    int max_level_;
    int max_level2_;
    int max_level3_;
    // temporal solution
    // 体素大小
    double voxel_size_;
    // 所有节点的信息
    std::vector<torch::Tensor> all_pts;
    std::vector<torch::Tensor> all_pts2;
    std::vector<torch::Tensor> all_pts3;
private:
    std::set<uint64_t> all_keys;
    std::set<uint64_t> all_keys2;
    std::set<uint64_t> all_keys3;

    // std::shared_ptr<Octant> root_;
    Octant *root_;
    Octant *root2_;
    Octant *root3_;
    // static int feature_index;

    // internal count function
    std::pair<int64_t, int64_t> count_nodes_internal();
    std::pair<int64_t, int64_t> count_recursive_internal(Octant *n);

// std::pair  只能存储两个元素 可以为不同类型  通过first和second访问
// std::tuple 可以存储多个元素 可以为不同类型  通过std::get函数来访问
};