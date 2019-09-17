#ifndef CK_TENSOR_COORDINATE_V2_HPP
#define CK_TENSOR_COORDINATE_V2_HPP

#include "common_header.hpp"
#include "dimension.hpp"
#include "multi_index_transform.hpp"
#include "tensor_descriptor.hpp"

namespace ck {

template <typename TensorDesc>
struct TensorCoordinate_v2;

template <typename NativeTensorDesc>
struct NativeTensorCoordinate
{
    using type                    = NativeTensorCoordinate;
    using tensor_desc_type        = NativeTensorDesc;
    static constexpr index_t nDim = tensor_desc_type::GetNumOfDimension();
    using Index                   = MultiIndex<nDim>;

    __host__ __device__ constexpr NativeTensorCoordinate(Index idx)
        : mIndex(idx), mOffset(tensor_desc_type::CalculateOffset(idx))
    {
    }

    template <typename... Xs>
    __host__ __device__ constexpr NativeTensorCoordinate(Xs... xs)
        : NativeTensorCoordinate(Index{xs...})
    {
    }

    template <index_t... Xs>
    __host__ __device__ constexpr NativeTensorCoordinate(Sequence<Xs...>)
        : NativeTensorCoordinate(Index{Xs...})
    {
    }

    __host__ __device__ static constexpr auto GetTensorDescriptor() { return tensor_desc_type{}; }

    __host__ __device__ constexpr const Index& GetIndex() const { return mIndex; }

    __host__ __device__ constexpr const index_t& GetOffset() const { return mOffset; }

    __host__ __device__ constexpr type operator+=(const Index& idx_diff)
    {
        // mIndex is updated here, but some (or all) of its entries may never be used
        mIndex += idx_diff;

        mOffset += tensor_desc_type::CalculateOffsetDiff(idx_diff);

        return *this;
    }

    __host__ __device__ constexpr type operator-=(const Index& idx_diff)
    {
        // mIndex is updated here, but some (or all) of its entries may never be used
        mIndex -= idx_diff;

        mOffset -= tensor_desc_type::CalculateOffsetDiff(idx_diff);

        return *this;
    }

    __host__ __device__ constexpr type operator+(const Index& idx_diff) const
    {
        type coord = *this;
        coord += idx_diff;
        return coord;
    }

    __host__ __device__ constexpr type operator-(const Index& idx_diff) const
    {
        type coord = *this;
        coord -= idx_diff;
        return coord;
    }

    // TODO: should this function be here? should it be specific for padding check?
    __host__ __device__ static constexpr bool IsAnyLevelIndexInPaddingArea() { return false; }

    private:
    // mIndex may be saved and update, however, the value of some (or all) of its entries may
    //   never be used. Compiler should be able to remove these entries as well as its calculation
    //   as dead code.
    // TODO: make sure compiler indeed remove these dead code
    Index mIndex;
    index_t mOffset;
};

template <typename TransformedTensorDesc>
struct TransformedTensorCoordinate
{
    using tensor_desc_type = TransformedTensorDesc;
    using LowerCoord =
        typename TensorCoordinate_v2<decltype(tensor_desc_type::GetLowerTensorDescriptor())>::type;
    using UpperCoord              = TransformedTensorCoordinate;
    static constexpr index_t nDim = tensor_desc_type::GetNumOfDimension();
    using UpperIndex              = MultiIndex<nDim>;

    __host__ __device__ constexpr TransformedTensorCoordinate(UpperIndex idx)
        : mIndexUp{idx}, mCoordLow{tensor_desc_type::CalculateLowerIndex(idx)}
    {
    }

    template <typename... Xs>
    __host__ __device__ constexpr TransformedTensorCoordinate(Xs... xs)
        : TransformedTensorCoordinate(UpperIndex{xs...})
    {
    }

    template <index_t... Xs>
    __host__ __device__ constexpr TransformedTensorCoordinate(Sequence<Xs...>)
        : TransformedTensorCoordinate(UpperIndex{Xs...})
    {
    }

    __host__ __device__ static constexpr auto GetTensorDescriptor() { return tensor_desc_type{}; }

    __host__ __device__ constexpr const LowerCoord& GetLowerCoordinate() const { return mCoordLow; }

    __host__ __device__ constexpr const UpperIndex& GetUpperIndex() const { return mIndexUp; }

    __host__ __device__ constexpr const UpperIndex& GetIndex() const { return GetUpperIndex(); }

    __host__ __device__ constexpr const index_t& GetOffset() const
    {
        return GetLowerCoordinate().GetOffset();
    }

    __host__ __device__ constexpr UpperCoord operator+=(const UpperIndex& idx_up_diff)
    {
        // For transformation of multi-index difference, not all transformation functions need to
        //   know the old lower-index or the old upper-index. We pass both of them to the
        //   transformation function. The transformation function itself decides to use them or not.
        mCoordLow += tensor_desc_type::CalculateLowerIndexDiff(
            idx_up_diff, GetIndex(), GetLowerCoordinate().GetIndex());

        // mIndexUp is updated here, but some (or all) of its entries may never be used
        mIndexUp += idx_up_diff;

        return *this;
    }

    __host__ __device__ constexpr UpperCoord operator-=(const UpperIndex& idx_up_diff)
    {
        mCoordLow -= tensor_desc_type::CalculateLowerIndexDiff(
            idx_up_diff, GetIndex(), GetLowerCoordinate().GetIndex());

        mIndexUp -= idx_up_diff;

        return *this;
    }

    __host__ __device__ constexpr UpperCoord operator+(const UpperIndex& idx_up_diff) const
    {
        UpperCoord coord_up = *this;
        coord_up += idx_up_diff;
        return coord_up;
    }

    __host__ __device__ constexpr UpperCoord operator-(const UpperIndex& idx_up_diff) const
    {
        UpperCoord coord_up = *this;
        coord_up -= idx_up_diff;
        return coord_up;
    }

    // TODO: should this function be here? should it be specific for padding check?
    __host__ __device__ constexpr bool IsAnyLevelIndexInPaddingArea() const
    {
        return tensor_desc_type::IsUpperIndexInPaddingArea(GetIndex()) ||
               mCoordLow.IsAnyLevelIndexInPaddingArea();
    }

    private:
    // mIndexUp may be calculated and update, however, the value of some (or all) of its entries may
    //   never be used. Compiler should be able to remove these entries as well as its calculation
    //   as dead code.
    // TODO: make sure compiler indeed remove these dead code
    UpperIndex mIndexUp;
    LowerCoord mCoordLow;
};

template <typename TensorDesc>
struct TensorCoordinate_v2
{
    private:
    template <typename... Ts>
    __host__ __device__ static constexpr auto
    MakeDummyTensorCoordinate(NativeTensorDescriptor<Ts...>)
    {
        return NativeTensorCoordinate<NativeTensorDescriptor<Ts...>>(
            make_zero_array<index_t, TensorDesc::GetNumOfDimension()>());
    }

    template <typename... Ts>
    __host__ __device__ static constexpr auto
    MakeDummyTensorCoordinate(TransformedTensorDescriptor<Ts...>)
    {
        return TransformedTensorCoordinate<TransformedTensorDescriptor<Ts...>>(
            make_zero_array<index_t, TensorDesc::GetNumOfDimension()>());
    }

    public:
    using type = decltype(MakeDummyTensorCoordinate(TensorDesc{}));
};

template <typename TensorDesc>
__host__ __device__ constexpr auto
make_tensor_coordinate_v2(TensorDesc, MultiIndex<TensorDesc::GetNumOfDimension()> idx)
{
    return typename TensorCoordinate_v2<TensorDesc>::type(idx);
}
}
#endif