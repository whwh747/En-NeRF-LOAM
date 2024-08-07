#include "../include/octree.h"
#include "../include/utils.h"
#include <queue>
#include <iostream>

// #define MAX_HIT_VOXELS 10
// #define MAX_NUM_VOXELS 10000

int Octant::next_index_ = 0;
// int Octree::feature_index = 0;

int tree1_index = 0;
int tree2_index = 0;
int tree3_index = 0;

int incr_x[8] = {0, 0, 0, 0, 1, 1, 1, 1};
int incr_y[8] = {0, 0, 1, 1, 0, 0, 1, 1};
int incr_z[8] = {0, 1, 0, 1, 0, 1, 0, 1};

Octree::Octree()
{
}

Octree::Octree(int64_t grid_dim, int64_t feat_dim, double voxel_size, std::vector<torch::Tensor> all_pts)
{
    Octant::next_index_ = 0;
    init(grid_dim, feat_dim, voxel_size);
    for (auto &pt : all_pts)
    {
        insert(pt);
    }
}

Octree::~Octree()
{
}

void Octree::init(int64_t grid_dim, int64_t feat_dim, double voxel_size)
{
    size_ = grid_dim;
    feat_dim_ = feat_dim;
    voxel_size_ = voxel_size;
    max_level_ = log2(size_);
    // root_ = std::make_shared<Octant>();
    root_ = new Octant();
    root_->side_ = size_;
    // root_->depth_ = 0;
    root_->is_leaf_ = false;
    root_->index_ = tree1_index++;

    size2_ = grid_dim/2;
    max_level2_ = log2(size2_);
    root2_ = new Octant();
    root2_->side_ = size2_;
    root2_->is_leaf_ = false;
    root2_->index_ = tree2_index++;

    size3_ = grid_dim/4;
    max_level3_ = log2(size3_);
    root3_ = new Octant();
    root3_->side_ = size3_;
    root3_->is_leaf_ = false;
    root3_->index_ = tree3_index++;


    // feats_allocated_ = 0;
    // auto options = torch::TensorOptions().requires_grad(true);
    // feats_array_ = torch::randn({MAX_NUM_VOXELS, feat_dim}, options) * 0.01;
}

void Octree::insert(torch::Tensor pts)
{
    // temporal solution
    all_pts.push_back(pts);

    if (root_ == nullptr)
    {
        std::cout << "Octree not initialized!" << std::endl;
    }

    auto points = pts.accessor<int, 2>();
    if (points.size(1) != 3)
    {
        std::cout << "Point dimensions mismatch: inputs are " << points.size(1) << " expect 3" << std::endl;
        return;
    }

    for (int i = 0; i < points.size(0); ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            int x = points[i][0] + incr_x[j];
            int y = points[i][1] + incr_y[j];
            int z = points[i][2] + incr_z[j];
            uint64_t key = encode(x, y, z);

            all_keys.insert(key);

            const unsigned int shift = MAX_BITS - max_level_ - 1;

            auto n = root_;
            unsigned edge = size_ / 2;
            for (int d = 1; d <= max_level_; edge /= 2, ++d)
            {
                const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
                // std::cout << "Level: " << d << " ChildID: " << childid << std::endl;
                auto tmp = n->child(childid);
                if (!tmp)
                {
                    const uint64_t code = key & MASK[d + shift];
                    const bool is_leaf = (d == max_level_);
                    // tmp = std::make_shared<Octant>();
                    tmp = new Octant();
                    tmp->index_ = tree1_index++;
                    tmp->code_ = code;
                    tmp->side_ = edge;
                    tmp->is_leaf_ = is_leaf;
                    tmp->type_ = is_leaf ? (j == 0 ? SURFACE : FEATURE) : NONLEAF;

                    n->children_mask_ = n->children_mask_ | (1 << childid);
                    n->child(childid) = tmp;
                }
                else
                {
                    if (tmp->type_ == FEATURE && j == 0)
                        tmp->type_ = SURFACE;
                }
                n = tmp;
            }
        }
    }
}

void Octree::insert2(torch::Tensor pts)
{
    all_pts2.push_back(pts);

    if (root2_ == nullptr)
    {
        std::cout << "Octree2 not initialized!" << std::endl;
    }

    auto points = pts.accessor<int, 2>();
    if (points.size(1) != 3)
    {
        std::cout << "Point dimensions mismatch: inputs are " << points.size(1) << " expect 3" << std::endl;
        return;
    }

    for (int i = 0; i < points.size(0); ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            int x = points[i][0] + incr_x[j];
            int y = points[i][1] + incr_y[j];
            int z = points[i][2] + incr_z[j];
            uint64_t key = encode(x, y, z);

            all_keys2.insert(key);

            const unsigned int shift = MAX_BITS - max_level2_ - 1;

            auto n = root2_;
            unsigned edge = size2_ / 2;
            for (int d = 1; d <= max_level2_; edge /= 2, ++d)
            {
                const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
                // std::cout << "Level: " << d << " ChildID: " << childid << std::endl;
                auto tmp = n->child(childid);
                if (!tmp)
                {
                    const uint64_t code = key & MASK[d + shift];
                    const bool is_leaf = (d == max_level2_);
                    tmp = new Octant();
                    tmp->index_ = tree2_index++;
                    tmp->code_ = code;
                    tmp->side_ = edge;
                    tmp->is_leaf_ = is_leaf;
                    tmp->type_ = is_leaf ? (j == 0 ? SURFACE : FEATURE) : NONLEAF;

                    n->children_mask_ = n->children_mask_ | (1 << childid);
                    n->child(childid) = tmp;
                }
                else
                {
                    if (tmp->type_ == FEATURE && j == 0)
                        tmp->type_ = SURFACE;
                }
                n = tmp;
            }

        }
    }
}

void Octree::insert3(torch::Tensor pts)
{
    all_pts3.push_back(pts);

    if (root3_ == nullptr)
    {
        std::cout << "Octree3 not initialized!" << std::endl;
    }

    auto points = pts.accessor<int, 2>();
    if (points.size(1) != 3)
    {
        std::cout << "Point dimensions mismatch: inputs are " << points.size(1) << " expect 3" << std::endl;
        return;
    }

    for (int i = 0; i < points.size(0); ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            int x = points[i][0] + incr_x[j];
            int y = points[i][1] + incr_y[j];
            int z = points[i][2] + incr_z[j];
            uint64_t key = encode(x, y, z);

            all_keys3.insert(key);

            const unsigned int shift = MAX_BITS - max_level3_ - 1;

            auto n = root3_;
            unsigned edge = size3_ / 2;
            for (int d = 1; d <= max_level3_; edge /= 2, ++d)
            {
                const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
                // std::cout << "Level: " << d << " ChildID: " << childid << std::endl;
                auto tmp = n->child(childid);
                if (!tmp)
                {
                    const uint64_t code = key & MASK[d + shift];
                    const bool is_leaf = (d == max_level3_);
                    tmp = new Octant();
                    tmp->index_ = tree3_index++;
                    tmp->code_ = code;
                    tmp->side_ = edge;
                    tmp->is_leaf_ = is_leaf;
                    tmp->type_ = is_leaf ? (j == 0 ? SURFACE : FEATURE) : NONLEAF;

                    n->children_mask_ = n->children_mask_ | (1 << childid);
                    n->child(childid) = tmp;
                }
                else
                {
                    if (tmp->type_ == FEATURE && j == 0)
                        tmp->type_ = SURFACE;
                }
                n = tmp;
            }

        }
    }
}

double Octree::try_insert(torch::Tensor pts)
{
    if (root_ == nullptr)
    {
        std::cout << "Octree not initialized!" << std::endl;
    }

    auto points = pts.accessor<int, 2>();
    if (points.size(1) != 3)
    {
        std::cout << "Point dimensions mismatch: inputs are " << points.size(1) << " expect 3" << std::endl;
        return -1.0;
    }

    std::set<uint64_t> tmp_keys;

    for (int i = 0; i < points.size(0); ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            int x = points[i][0] + incr_x[j];
            int y = points[i][1] + incr_y[j];
            int z = points[i][2] + incr_z[j];
            uint64_t key = encode(x, y, z);

            tmp_keys.insert(key);
        }
    }

    std::set<int> result;
    std::set_intersection(all_keys.begin(), all_keys.end(),
                          tmp_keys.begin(), tmp_keys.end(),
                          std::inserter(result, result.end()));

    double overlap_ratio = 1.0 * result.size() / tmp_keys.size();
    return overlap_ratio;
}

Octant *Octree::find_octant(std::vector<float> coord)
{
    int x = int(coord[0]);
    int y = int(coord[1]);
    int z = int(coord[2]);
    // uint64_t key = encode(x, y, z);
    // const unsigned int shift = MAX_BITS - max_level_ - 1;

    auto n = root_;
    unsigned edge = size_ / 2;
    for (int d = 1; d <= max_level_; edge /= 2, ++d)
    {
        const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
        auto tmp = n->child(childid);
        if (!tmp)
            return nullptr;

        n = tmp;
    }
    return n;
}
Octant *Octree::find_octant_fa(std::vector<float> coord)
{
    int x = int(coord[0]);
    int y = int(coord[1]);
    int z = int(coord[2]);
    // uint64_t key = encode(x, y, z);
    // const unsigned int shift = MAX_BITS - max_level_ - 1;

    auto n2 = root2_;
    unsigned edge = size2_ / 2;
    for (int d = 1; d <= max_level2_; edge /= 2, ++d)
    {
        const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
        auto tmp = n2->child(childid);
        if (!tmp)
            return nullptr;

        n2 = tmp;
    }
    return n2;
}

Octant *Octree::find_octant_fafa(std::vector<float> coord)
{
    int x = int(coord[0]);
    int y = int(coord[1]);
    int z = int(coord[2]);
    // uint64_t key = encode(x, y, z);
    // const unsigned int shift = MAX_BITS - max_level_ - 1;

    auto n3 = root3_;
    unsigned edge = size3_ / 2;
    for (int d = 1; d <= max_level3_; edge /= 2, ++d)
    {
        const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
        auto tmp = n3->child(childid);
        if (!tmp)
            return nullptr;

        n3 = tmp;
    }
    return n3;
}

bool Octree::has_voxel(torch::Tensor pts)
{
    if (root_ == nullptr)
    {
        std::cout << "Octree not initialized!" << std::endl;
    }

    auto points = pts.accessor<int, 1>();
    if (points.size(0) != 3)
    {
        return false;
    }

    int x = int(points[0]);
    int y = int(points[1]);
    int z = int(points[2]);

    auto n = root_;
    unsigned edge = size_ / 2;
    for (int d = 1; d <= max_level_; edge /= 2, ++d)
    {
        const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
        auto tmp = n->child(childid);
        if (!tmp)
            return false;

        n = tmp;
    }

    if (!n)
        return false;
    else
        return true;
}

torch::Tensor Octree::get_features(torch::Tensor pts)
{
}

torch::Tensor Octree::get_leaf_voxels()
{
    std::vector<float> voxel_coords = get_leaf_voxel_recursive(root_);

    int N = voxel_coords.size() / 3;
    torch::Tensor voxels = torch::from_blob(voxel_coords.data(), {N, 3});
    return voxels.clone();
}

std::vector<float> Octree::get_leaf_voxel_recursive(Octant *n)
{
    if (!n)
        return std::vector<float>();

    if (n->is_leaf_ && n->type_ == SURFACE)
    {
        auto xyz = decode(n->code_);
        return {xyz[0], xyz[1], xyz[2]};
    }

    std::vector<float> coords;
    for (int i = 0; i < 8; i++)
    {
        auto temp = get_leaf_voxel_recursive(n->child(i));
        coords.insert(coords.end(), temp.begin(), temp.end());
    }

    return coords;
}

torch::Tensor Octree::get_voxels()
{
    std::vector<float> voxel_coords = get_voxel_recursive(root_);
    int N = voxel_coords.size() / 4;
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor voxels = torch::from_blob(voxel_coords.data(), {N, 4}, options);
    return voxels.clone();
}

std::vector<float> Octree::get_voxel_recursive(Octant *n)
{
    if (!n)
        return std::vector<float>();

    auto xyz = decode(n->code_);
    std::vector<float> coords = {xyz[0], xyz[1], xyz[2], float(n->side_)};
    for (int i = 0; i < 8; i++)
    {
        auto temp = get_voxel_recursive(n->child(i));
        coords.insert(coords.end(), temp.begin(), temp.end());
    }

    return coords;
}

std::pair<int64_t, int64_t> Octree::count_nodes_internal()
{
    return count_recursive_internal(root_);
}

// int64_t Octree::leaves_count_recursive(std::shared_ptr<Octant> n)
std::pair<int64_t, int64_t> Octree::count_recursive_internal(Octant *n)
{
    if (!n)
        return std::make_pair<int64_t, int64_t>(0, 0);

    if (n->is_leaf_)
        return std::make_pair<int64_t, int64_t>(1, 1);

    auto sum = std::make_pair<int64_t, int64_t>(1, 0);

    for (int i = 0; i < 8; i++)
    {
        auto temp = count_recursive_internal(n->child(i));
        sum.first += temp.first;
        sum.second += temp.second;
    }

    return sum;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Octree::get_centres_and_children()
{
    auto node_count = count_nodes_internal();
    auto total_count = node_count.first;
    auto leaf_count = node_count.second;
    // std::cout<<"total_count = "<<total_count<<std::endl;
    // std::cout<<"size_ = "<<size_<<std::endl;
    auto all_voxels = torch::zeros({total_count, 4}, dtype(torch::kFloat32));
    auto all_children = -torch::ones({total_count, 8}, dtype(torch::kFloat32));
    auto all_features = -torch::ones({total_count, 8}, dtype(torch::kInt32));



    std::queue<Octant *> all_nodes;
    all_nodes.push(root_);

    while (!all_nodes.empty())
    {
        auto node_ptr = all_nodes.front();
        all_nodes.pop();

        auto xyz = decode(node_ptr->code_);
        std::vector<float> coords = {xyz[0], xyz[1], xyz[2], float(node_ptr->side_)};
        auto voxel = torch::from_blob(coords.data(), {4}, dtype(torch::kFloat32));
        all_voxels[node_ptr->index_] = voxel;

        if (node_ptr->type_ == SURFACE)
        {
            for (int i = 0; i < 8; ++i)
            {
                std::vector<float> vcoords = coords;
                vcoords[0] += incr_x[i];
                vcoords[1] += incr_y[i];
                vcoords[2] += incr_z[i];
                auto voxel = find_octant(vcoords);
                if (voxel)
                    all_features.data_ptr<int>()[node_ptr->index_ * 8 + i] = voxel->index_;
            }
        }

        for (int i = 0; i < 8; i++)
        {
            auto child_ptr = node_ptr->child(i);
            if (child_ptr && child_ptr->type_ != FEATURE)
            {
                all_nodes.push(child_ptr);
                all_children[node_ptr->index_][i] = float(child_ptr->index_);
            }
        }
    }
    auto all_voxels2 = torch::zeros({total_count, 4}, dtype(torch::kFloat32));
    auto all_children2 = -torch::ones({total_count, 8}, dtype(torch::kFloat32));
    auto all_features_fa = -torch::ones({total_count, 8}, dtype(torch::kInt32));
    std::queue<Octant *> all_nodes2;
    all_nodes2.push(root2_);

    while (!all_nodes2.empty())
    {
        auto node_ptr = all_nodes2.front();
        all_nodes2.pop();

        auto xyz = decode(node_ptr->code_);
        std::vector<float> coords = {xyz[0], xyz[1], xyz[2], float(node_ptr->side_)};
        auto voxel = torch::from_blob(coords.data(), {4}, dtype(torch::kFloat32));
        all_voxels2[node_ptr->index_] = voxel;

        if (node_ptr->type_ == SURFACE)
        {
            for (int i = 0; i < 8; ++i)
            {
                std::vector<float> vcoords = coords;
                vcoords[0] += incr_x[i];
                vcoords[1] += incr_y[i];
                vcoords[2] += incr_z[i];
                auto voxel2 = find_octant_fa(vcoords);
                if(voxel2)
                    all_features_fa.data_ptr<int>()[node_ptr->index_ * 8 + i] = voxel2->index_;
            }
        }

        for (int i = 0; i < 8; i++)
        {
            auto child_ptr = node_ptr->child(i);
            if (child_ptr && child_ptr->type_ != FEATURE)
            {
                all_nodes2.push(child_ptr);
                all_children2[node_ptr->index_][i] = float(child_ptr->index_);
            }
        }
    }

    auto all_voxels3 = torch::zeros({total_count, 4}, dtype(torch::kFloat32));
    auto all_children3 = -torch::ones({total_count, 8}, dtype(torch::kFloat32));
    auto all_features_fafa = -torch::ones({total_count, 8}, dtype(torch::kInt32));
    std::queue<Octant *> all_nodes3;
    all_nodes3.push(root3_);

    while (!all_nodes3.empty())
    {
        auto node_ptr = all_nodes3.front();
        all_nodes3.pop();

        auto xyz = decode(node_ptr->code_);
        std::vector<float> coords = {xyz[0], xyz[1], xyz[2], float(node_ptr->side_)};
        auto voxel = torch::from_blob(coords.data(), {4}, dtype(torch::kFloat32));
        all_voxels3[node_ptr->index_] = voxel;

        if (node_ptr->type_ == SURFACE)
        {
            for (int i = 0; i < 8; ++i)
            {
                std::vector<float> vcoords = coords;
                vcoords[0] += incr_x[i];
                vcoords[1] += incr_y[i];
                vcoords[2] += incr_z[i];
                auto voxel3 = find_octant_fafa(vcoords);
                if(voxel3)
                    all_features_fafa.data_ptr<int>()[node_ptr->index_ * 8 + i] = voxel3->index_;
            }
        }

        for (int i = 0; i < 8; i++)
        {
            auto child_ptr = node_ptr->child(i);
            if (child_ptr && child_ptr->type_ != FEATURE)
            {
                all_nodes3.push(child_ptr);
                all_children3[node_ptr->index_][i] = float(child_ptr->index_);
            }
        }
    }
    // return std::make_tuple(all_voxels, all_children, all_features);
    return std::make_tuple(all_voxels, all_children, all_features, all_features_fa, all_features_fafa);
}

int64_t Octree::count_nodes()
{
    return count_recursive(root_);
}

// int64_t Octree::leaves_count_recursive(std::shared_ptr<Octant> n)
int64_t Octree::count_recursive(Octant *n)
{
    if (!n)
        return 0;

    int64_t sum = 1;

    for (int i = 0; i < 8; i++)
    {
        sum += count_recursive(n->child(i));
    }

    return sum;
}

int64_t Octree::count_leaf_nodes()
{
    return leaves_count_recursive(root_);
}

// int64_t Octree::leaves_count_recursive(std::shared_ptr<Octant> n)
int64_t Octree::leaves_count_recursive(Octant *n)
{
    if (!n)
        return 0;

    if (n->type_ == SURFACE)
    {
        return 1;
    }

    int64_t sum = 0;

    for (int i = 0; i < 8; i++)
    {
        sum += leaves_count_recursive(n->child(i));
    }

    return sum;
}
