#include <iostream>  // NOLINT

#include <gtest/gtest.h>

#include "../include/voxblox_fast/core/object_pool.h"

using namespace voxblox_fast;  // NOLINT

TEST(MemoryPool, AllocateGet) {
  struct Abc {
    Abc(int _a , int _b, int _c) : a(_a), b(_b), c(_c) {
      std::cout << "ctor: " << a << "," << b << "," << c << std::endl;
    }
    ~Abc() {
      std::cout << "dtor: " << a << "," << b << "," << c << std::endl;
    }
    int a, b, c;
  };

  ObjectPool<Abc> pool;
  Abc* abc = pool.AllocateObject(10 , 20 , 30);
  EXPECT_EQ(abc->a, 10);
  EXPECT_EQ(abc->b, 20);
  EXPECT_EQ(abc->c, 30);

  Abc* abc2 = pool.AllocateObject(40 , 50 , 60);
  EXPECT_EQ(abc->a, 10);
  EXPECT_EQ(abc->b, 20);
  EXPECT_EQ(abc->c, 30);
  EXPECT_EQ(abc2->a, 40);
  EXPECT_EQ(abc2->b, 50);
  EXPECT_EQ(abc2->c, 60);
}

// TODO(schneith): Fix the impl hack with AllocateObjectShared.
TEST(MemoryPool, DISABLE_AllocateSharedReleasing) {
  struct Char {
    Char(char _data) : data(_data) {
      std::cout << "ctor: " << data << std::endl;
    }
    ~Char() {
      std::cout << "dtor: " << data << std::endl;
    }
    char data;
  };
  typedef std::shared_ptr<Char> CharPtr;

  ObjectPool<Char, /*kNumAlignedBytes=*/1, /*kMemoryChunkSizeBytes=*/2> pool;
  EXPECT_EQ(pool.GetNumMemoryChunks(), 0u);

  {
    CharPtr a = pool.AllocateObjectShared('a');
    EXPECT_EQ(pool.GetNumMemoryChunks(), 1u);
    EXPECT_EQ(a->data, 'a');
  }
  // CharPtr a went out of scope and the memory chunk should have been released.
  EXPECT_EQ(pool.GetNumMemoryChunks(), 0u);
}

TEST(MemoryPool, DeAllocateMemoryChunks) {
  ObjectPool<char, /*kNumAlignedBytes=*/1, /*kMemoryChunkSizeBytes=*/2> pool;

  // Let's add 2 chars to fill up the first chunk.
  EXPECT_EQ(pool.GetNumMemoryChunks(), 0u);
  char* a = pool.AllocateObject('a');
  EXPECT_EQ(pool.GetNumMemoryChunks(), 1u);
  char* b = pool.AllocateObject('b');
  EXPECT_EQ(pool.GetNumMemoryChunks(), 1u);

  EXPECT_EQ(*a, 'a');
  EXPECT_EQ(*b, 'b');

  // The next 5 slots require a new chunk.
  char* c = pool.AllocateObject('c');
  EXPECT_EQ(pool.GetNumMemoryChunks(), 2u);
  char* d = pool.AllocateObject('d');
  EXPECT_EQ(pool.GetNumMemoryChunks(), 2u);
  EXPECT_EQ(*a, 'a');
  EXPECT_EQ(*b, 'b');
  EXPECT_EQ(*c, 'c');
  EXPECT_EQ(*d, 'd');

  // Remove all elements of the first chunk. This should deallcoate chunk 0.
  pool.DeallocateObject(a);
  EXPECT_EQ(pool.GetNumMemoryChunks(), 2u);
  pool.DeallocateObject(b);
  EXPECT_EQ(pool.GetNumMemoryChunks(), 1u);
  EXPECT_EQ(*c, 'c');
  EXPECT_EQ(*d, 'd');

  // Lets add one more.
  char* e = pool.AllocateObject('e');
  EXPECT_EQ(pool.GetNumMemoryChunks(), 2u);
  char* f = pool.AllocateObject('f');
  EXPECT_EQ(pool.GetNumMemoryChunks(), 2u);
  EXPECT_EQ(*c, 'c');
  EXPECT_EQ(*d, 'd');
  EXPECT_EQ(*e, 'e');
  EXPECT_EQ(*f, 'f');
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  int result = RUN_ALL_TESTS();
  return result;
}
