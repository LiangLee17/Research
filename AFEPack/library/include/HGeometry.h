/**
 * @file   HGeometry.h
 * @author Robert Lie
 * @date   Sun Apr 29 11:06:40 2007
 * 
 * @brief  
 * 
 * 
 */

#ifndef _HGeometry_h_
#define _HGeometry_h_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <list>
#include <iterator>
#include <algorithm>

#include <base/exceptions.h>

#include "DerefIterator.h"
#include "Geometry.h"
#include "TemplateElement.h"
#include "PropertyTable.h"

AFEPACK_OPEN_NAMESPACE

template <int DIM, int DOW> class HGeometry;
template <int DOW> class HGeometry<0,DOW>;
template <int DOW> class HGeometry<1,DOW>;
template <int DOW> class HGeometry<2,DOW>;
template <int DOW> class HGeometry<3,DOW>;
template <int DIM, int DOW> class HGeometryTree;
template <int DIM, int DOW> class RegularMesh;
template <int DIM, int DOW> class IrregularMesh;
template <int DIM, int DOW> class HElement;

template <int DIM, int DOW> class ElementIterator;
template <int DIM, int DOW> class RootFirstElementIterator;
template <int DIM, int DOW> class ActiveElementIterator;
template <int DIM, int DOW> class IrregularMeshPair;
template <int DIM, int DOW> class ActiveElementPairIterator;

template <int DIM, int DOW> std::ostream& operator<<(std::ostream&, const HGeometry<DIM,DOW>&);
template <int DIM, int DOW> std::ostream& operator<<(std::ostream&, const HElement<DIM, DOW>&);

template <int DIM, int DOW> std::ostream& operator<<(std::ostream&, IrregularMesh<DIM, DOW>&);
template <int DOW> std::ostream& operator<<(std::ostream&, const HGeometry<0,DOW>&);

template <int DIM, int DOW> bool operator==(const ElementIterator<DIM, DOW>&,
                                            const ElementIterator<DIM, DOW>&);
template <int DIM, int DOW> bool operator!=(const ElementIterator<DIM, DOW>&,
                                            const ElementIterator<DIM, DOW>&);
template <int DIM, int DOW> bool operator==(const ElementIterator<DIM, DOW>&,
                                            ElementIterator<DIM, DOW>&);
template <int DIM, int DOW> bool operator!=(const ElementIterator<DIM, DOW>&,
                                            ElementIterator<DIM, DOW>&);

template <int DIM, int DOW> bool operator==(const ActiveElementPairIterator<DIM, DOW>&,
                                            const ActiveElementPairIterator<DIM, DOW>&);
template <int DIM, int DOW> bool operator!=(const ActiveElementPairIterator<DIM, DOW>&,
                                            const ActiveElementPairIterator<DIM, DOW>&);
template <int DIM, int DOW> bool operator==(const ActiveElementPairIterator<DIM, DOW>&,
                                            ActiveElementPairIterator<DIM, DOW>&);
template <int DIM, int DOW> bool operator!=(const ActiveElementPairIterator<DIM, DOW>&,
                                            ActiveElementPairIterator<DIM, DOW>&);

template <int DIM>
struct HGeometryInfo {
  enum {
    dimension = DIM,
    n_vertex = DIM + 1,
    n_boundary = DIM + 1,
    n_child = (1<<DIM)
  };
};

template <> 
struct HGeometryInfo<0> {
  enum {
    dimension = 0,
    n_vertex = 0,
    n_boundary = 0,
    n_child = 0
  };
};

template <> 
struct HGeometryInfo<1> {
  enum {
    dimension = 1,
    n_vertex = 2,
    n_boundary = 0,
    n_child = 2
  };
};

typedef int bmark_t;

#ifdef __SERIALIZATION__
#include "Migration.h"
struct HGeometryBase : public PropertyTable, public AFEPack::Migration::HBuffer
#else
struct HGeometryBase : public PropertyTable
#endif // __SERIALIZATION__
{
  virtual ~HGeometryBase() {};
};

/**
 * ??????????????????????????????????????????????????????????????????????????? index ???????????????
 * USED?????????????????????????????? UNUSED??????????????????????????????????????????????????????
 * ???????????????????????????????????????????????????????????????????????????
 *
 * ?????????????????????????????????????????????????????? ACTIVE ??? INACTIVE?????????
 * ACTIVE ??? USED ??????????????????????????????????????????????????????????????????????????????
 * ???????????????????????????????????????????????? INACTIVE ???????????????
 */
struct HTools {

  enum { USED = -8, UNUSED = -7,
         INACTIVE = -9, ACTIVE = USED };

  //@{
  /**
   * ???????????????????????????????????? index ?????????????????????????????????
   * 
   */

  /**
   * ???????????????????????????????????????
   */
  template <class GEO> bool
  isGeometryUsed(const GEO& geo) const {
    return (geo.index == USED);
  }

  /**
   * ????????????????????????????????????
   */
  template <class GEO> bool
  isGeometryActive(const GEO& geo) const {
    return (geo.index == ACTIVE);
  }

  /**
   * ???????????????????????????????????????
   */
  template <class GEO> bool
  isGeometryInactive(const GEO& geo) const {
    return (geo.index == INACTIVE);
  }

  /**
   * ????????????????????????????????????
   */
  template <class GEO> bool
  isGeometryIndexed(const GEO& geo) const {
    return (geo.index >= 0);
  }
  //@}

  //@{
  /**
   * ???????????????????????????????????????????????????
   */

  //@{
  /**
   * ?????????????????????????????? USED???
   */
  template <class GEO> void 
  setGeometryUsed(GEO& geo) const {
    geo.index = USED;
    for (u_int i = 0;i < GEO::n_boundary;++ i) {
      this->setGeometryUsed(*geo.boundary[i]);
    }
  }
  template <int DOW> void 
  setGeometryUsed(HGeometry<0,DOW>& geo) const {
    geo.index = USED;
  }
  template <int DOW> void 
  setGeometryUsed(HGeometry<1,DOW>& geo) const {
    geo.index = USED;
  }
  template <class GEO> void 
  setGeometryActive(GEO& geo) const { /// USED == ACTIVE
    setGeometryUsed(geo);
  }
  //@}

  //@{
  /**
   * ???????????????????????????????????????????????????????????????????????????????????????????????????
   */
  template <class GEO> void 
  setGeometryUnused(GEO& geo) const {
    geo.index = UNUSED;
    for (u_int i = 0;i < GEO::n_boundary;++ i) {
      this->setGeometryUnused(*geo.boundary[i]);
    }
  }
  template <int DOW> void 
  setGeometryUnused(HGeometry<1,DOW>& geo) const {
    geo.index = UNUSED;
  }
  //@}

  //@{
  /**
   * ???????????????????????????????????????????????????????????????????????????????????????????????????
   */
  template <class GEO> void 
  setGeometryInactive(GEO& geo) const {
    geo.index = INACTIVE;
    for (u_int i = 0;i < GEO::n_boundary;++ i) {
      this->setGeometryInactive(*geo.boundary[i]);
    }
  }
  template <int DOW> void 
  setGeometryInactive(HGeometry<1,DOW>& geo) const {
    geo.index = INACTIVE;
  }
  //@}

  /**
   * ???????????????????????????????????????????????????
   */
  template <class GEO> void 
  setGeometryUnusedRecursively(GEO& geo) const {
    setGeometryUnused(geo);

    if (geo.isRefined()) {
      for (u_int i = 0;i < GEO::n_child;++ i) {
        this->setGeometryUnusedRecursively(*geo.child[i]);
      }
    }
  }
  //@}

  /**
   * ???????????????????????????????????????
   */
  template <class GEO> bool 
  isRefined(const GEO& geo) const {
    if (geo.isRefined()) {
      for (int n = 0;n < geo.n_child;++ n) {
        if (this->isGeometryUsed(*geo.child[n])) {
          return true;
        }
      }
    }

    return false;
  }

  /**
   * ???????????????????????????????????????????????????????????????????????????????????????????????????
   * ??????????????????????????????????????????????????????????????????????????????????????????????????????
   * ??????????????????false???????????????true???????????????????????????????????????????????????
   * ?????????????????????????????????????????? 1???
   */
  template <int DOW> bool 
  isSemiregular(const HGeometry<1,DOW>& geo) const {
    assert (this->isGeometryUsed(geo));

    bool result = true;
    if (geo.isRefined()) {
      if (this->isRefined(*geo.child[0]) ||
          this->isRefined(*geo.child[1])) {
        result = false;
      }
    }
    return result;
  }

  /**
   * ???????????????????????????????????????????????????????????????????????????
   *
   *   - ?????????????????????????????????
   *   - ?????????????????????????????? 1???
   *
   * ???????????????????????????????????????????????????????????? 
   */
  template <int DOW> bool 
  isSemiregular(const HGeometry<2,DOW>& geo) const {
    assert (this->isGeometryUsed(geo));

    u_int n_refined_edge = 0;
    for (u_int i = 0;i < geo.n_boundary;++ i) {
      const HGeometry<1,DOW>& edge = *geo.boundary[i];
      if(! this->isSemiregular(edge)) return false;
      if (this->isRefined(edge)) {
        n_refined_edge += 1;
      }
    }
    bool result = (n_refined_edge <= 1);
    return result;
  }

  template <class HGEO> bool
  semiregularizeBoundary(HGEO& geo) const {
    return false;
  }

  template <int DOW> bool
  semiregularizeBoundary(HGeometry<3,DOW>& geo) const {
    assert (this->isGeometryUsed(geo));

    bool result = false;
    for (int i = 0;i < geo.n_boundary;++ i) {
      HGeometry<2,DOW>& face = *geo.boundary[i];
      u_int n_refined_face_edge = 0;
      for (int j = 0;j < face.n_boundary;++ j) {
        const HGeometry<1,DOW> * edge = face.boundary[j];
        if (this->isRefined(*edge)) {
          n_refined_face_edge += 1;
        }
      }

      bool is_operated = false;
      if (n_refined_face_edge >= 2) {
        if (! face.isRefined()) face.refine();
        
        for (int j = 0;j < face.n_child;++ j) {
          HGeometry<2,DOW>& chd = *(face.child[j]);
          if (this->isGeometryUsed(chd)) continue;

          this->setGeometryUsed(chd);
          is_operated = true;
        }
      }
      if (is_operated) {
	i = -1;
	result = true;
      }
    }
    return result;
  }

  /**
   * ???????????????????????????????????????????????????????????????????????????????????????????????????
   * ???????????????????????????????????????????????????????????????????????????????????????????????????
   * ???????????????????????????????????????????????????????????????????????????????????????????????????
   * ??????????????????bug???????????????????????????????????????????????????????????????????????????
   * ???????????????????????????????????????
   */
  template <int DOW> bool 
  isSemiregular(const HGeometry<3,DOW>& geo) const {
    assert (this->isGeometryUsed(geo));

    u_int n_refined_edge = 0;
    u_int n_refined_face = 0;
    for (int i = 0;i < geo.n_boundary;++ i) {
      const HGeometry<2,DOW>& face = *geo.boundary[i];
      u_int n_refined_face_edge = 0;
      for (int j = 0;j < face.n_boundary;++ j) {
        const HGeometry<1,DOW> * edge = face.boundary[j];
        if (! this->isSemiregular(*edge)) return false;
        if (this->isRefined(*edge)) {
          n_refined_face_edge += 1;
        }
      }

      if (n_refined_face_edge == 3) { /// ??????????????????????????????
        n_refined_face += 1;
        /// ???????????????????????????????????????????????????????????????
        const HGeometry<2,DOW> * chd3 = face.child[3];
        for (int j = 0;j < chd3->n_boundary;++ j) {
          const HGeometry<1,DOW> * edge = chd3->boundary[j];
          if (this->isRefined(*edge)) return false;
        }
      }

      n_refined_edge += n_refined_face_edge;
    }
    if (n_refined_edge%2 != 0) abort();
    n_refined_edge /= 2; // ???????????????????????????
    bool result = (n_refined_edge <= 1 || 
                   (n_refined_edge == 3 && 
                    n_refined_face == 1));
    return result;
  }

  /**
   * ????????????????????????????????? INACTIVE ?????????
   */
  template <class GEO> void
  setParentInactive(GEO& geo) const {
    if (geo.parent == NULL) return;
    setGeometryInactive(*geo.parent);
  }

  //@{
  /**
   * ???????????????????????????????????????????????????????????????????????????????????????????????????
   *
   *   - ????????????????????? index ?????? 0???
   *   - ???????????????????????????????????????????????????????????? index ?????? 1???
   *   - ??????????????????????????????????????????????????? index ??? 0????????????????????????
   *     ?????????????????????index ?????? 1;
   */

  //@{
  /**
   * ??????????????? index ?????? 0.
   */
  template <class GEO> void
  clearIndex(GEO& geo) const {
    geo.index = 0;
    for (u_int i = 0;i < geo.n_boundary;++ i) {
      clearIndex(*geo.boundary[i]);
    }
    if (geo.isRefined()) {
      for (u_int i = 0;i < geo.n_child;++ i) {
        clearIndex(*geo.child[i]);
      }
    }
  }
  template <int DOW> void
  clearIndex(HGeometry<0,DOW>& geo) const {
    geo.index = 0;
  }
  template <int DOW> void
  clearIndex(HGeometry<1,DOW>& geo) const {
    geo.index = 0;
    for (u_int i = 0;i < geo.n_vertex;++ i) {
      clearIndex(*geo.vertex[i]);
    }
    if (geo.isRefined()) {
      for (u_int i = 0;i < geo.n_child;++ i) {
        clearIndex(*geo.child[i]);
      }
    }
  }
  //@}

  //@{
  /**
   * ?????????????????? index???
   */
  template <class GEO> void
  incrIndex(GEO& geo) const {
    geo.index += 1;
    for (u_int i = 0;i < geo.n_boundary;++ i) {
      incrIndex(*geo.boundary[i]);
    }
    if (geo.isRefined()) {
      for (u_int i = 0;i < geo.n_child;++ i) {
        incrIndex(*geo.child[i]);
      }
    }
  }
  template <int DOW> void
  incrIndex(HGeometry<0,DOW>& geo) const {
    geo.index += 1;
  }
  template <int DOW> void
  incrIndex(HGeometry<1,DOW>& geo) const {
    geo.index += 1;
    for (u_int i = 0;i < geo.n_vertex;++ i) {
      incrIndex(*geo.vertex[i]);
    }
    if (geo.isRefined()) {
      for (u_int i = 0;i < geo.n_child;++ i) {
        incrIndex(*geo.child[i]);
      }
    }
  }
  //@}

  //@{
  /**
   * ?????????????????? index???
   */
  template <class GEO> void
  decrIndex(GEO& geo) const {
    geo.index -= 1;
    if (geo.isRefined()) {
      for (u_int i = 0;i < geo.n_child;++ i) {
        decrIndex(*geo.child[i]);
      }
    }
    for (u_int i = 0;i < geo.n_boundary;++ i) {
      decrIndex(*geo.boundary[i]);
    }
    if (geo.index == 0) delete &geo;
  }
  template <int DOW> void
  decrIndex(HGeometry<0,DOW>& geo) const {
    geo.index -= 1;
    if (geo.index == 0) delete &geo;
  }
  template <int DOW> void
  decrIndex(HGeometry<1,DOW>& geo) const {
    geo.index -= 1;
    if (geo.isRefined()) {
      for (u_int i = 0;i < geo.n_child;++ i) {
        decrIndex(*geo.child[i]);
      }
    }
    for (u_int i = 0;i < geo.n_vertex;++ i) {
      decrIndex(*geo.vertex[i]);
    }
    if (geo.index == 0) delete &geo;
  }
  //@}
  //@}

  //@{
  template <class HGEO, class VTX> void
  regularize_add_node(HGEO& hgeo, 
                      GeometryBM& geo,
                      VTX& vtx) const {
    int idx = hgeo.index;
    if (geo.index() == -1) {
      vtx = hgeo; /// ??????????????????

      geo.index() = idx;
      geo.vertex().resize(1, idx);
      geo.boundary().resize(1, idx);
      geo.boundaryMark() = hgeo.bmark;
    }
  }

  template <class HGEO> void
  regularize_add_side(HGEO& hgeo, 
                      GeometryBM& geo) const {
    int idx = hgeo.index;
    if (geo.index() == -1) {
      geo.index() = idx;

      geo.vertex().resize(2);
      geo.vertex(0) = hgeo.vertex[0]->index;
      geo.vertex(1) = hgeo.vertex[1]->index;

      geo.boundary().resize(2);
      geo.boundary(0) = hgeo.vertex[0]->index;
      geo.boundary(1) = hgeo.vertex[1]->index;

      geo.boundaryMark() = hgeo.bmark;
    }
  }

  template <class HGEO> void
  regularize_add_triangle(HGEO& hgeo, 
                          GeometryBM& geo) const {
    int idx = hgeo.index;
    if (geo.index() == -1) {
      geo.index() = idx;

      geo.vertex().resize(3);
      geo.vertex(0) = hgeo.vertex[0]->index;
      geo.vertex(1) = hgeo.vertex[1]->index;
      geo.vertex(2) = hgeo.vertex[2]->index;

      geo.boundary().resize(3);
      geo.boundary(0) = hgeo.boundary[0]->index;
      geo.boundary(1) = hgeo.boundary[1]->index;
      geo.boundary(2) = hgeo.boundary[2]->index;

      geo.boundaryMark() = hgeo.bmark;
    }
  }

  template <class HGEO> void
  regularize_add_twin_triangle(HGEO& hgeo, 
                               GeometryBM& geo, 
                               int k) const {
    int idx = hgeo.index;
    int ii[] = {0,1,2,0,1,2,0,1,2};
    if (geo.index() == -1) {
      geo.index() = idx;

      geo.vertex().resize(4);
      geo.vertex(0) = hgeo.vertex[k]->index;
      geo.vertex(1) = hgeo.vertex[ii[k + 1]]->index;
      geo.vertex(2) = hgeo.boundary[k]->child[0]->vertex[1]->index;
      geo.vertex(3) = hgeo.vertex[ii[k + 2]]->index;

      geo.boundary().resize(4);
      geo.boundary(0) = hgeo.boundary[ii[k + 2]]->index;
      if (hgeo.boundary[k]->child[0]->vertex[0] == hgeo.vertex[ii[k + 1]]) {
	geo.boundary(1) = hgeo.boundary[k]->child[0]->index;
	geo.boundary(2) = hgeo.boundary[k]->child[1]->index;
      } else {
	geo.boundary(1) = hgeo.boundary[k]->child[1]->index;
	geo.boundary(2) = hgeo.boundary[k]->child[0]->index;
      }
      geo.boundary(3) = hgeo.boundary[ii[k + 1]]->index;

      geo.boundaryMark() = hgeo.bmark;
    }
  }
  //@}
};

/**
 * Hierarchy geometry. This is the basis class to make the hierarchy geometry
 * to be able to refined. It store the information of the realation ship of the
 * hierarchy tree structure.
 */
template <int DIM, int DOW=DIM>
  class HGeometry : public HGeometryInfo<DIM>, public HGeometryBase
  {
  public:
  enum { dim = DIM, dow = DOW };
  typedef HGeometry<0,dow> vertex_t;
  typedef HGeometry<dim-1,dow> bound_t;
  typedef HGeometry<dim,dow> this_t;
  typedef this_t child_t;
  typedef this_t parent_t;

  int index;
  std::vector<vertex_t *> vertex;
  std::vector<bound_t *> boundary;
  parent_t * parent;
  std::vector<child_t *> child;
  bmark_t bmark;

  public:
  HGeometry();
  virtual ~HGeometry() {}

  public:
  bool isRefined() const;
  bool isIncludePoint(const Point<DOW>&) const;
  void refine();
  void checkIntegrity() const;

  friend std::ostream& operator<< <>(std::ostream&, const HGeometry<DIM,DOW>&);
  };

/**
 * 0 dimensional hierarchy geometry. It's special because it's in fact is not a
 * in a hierarchy tree, instead there are the information of the coordinate of the
 * point.
 */
template <int DOW>
class HGeometry<0,DOW> : public Point<DOW>, public HGeometryInfo<0>, public HGeometryBase
{
 public:
  enum { dim = 0, dow = DOW };
  typedef HGeometry<dim,dow> this_t;
  typedef this_t vertex_t;
  typedef this_t bound_t;
  typedef this_t child_t;
  typedef this_t parent_t;

  int index;
  bmark_t bmark;

  static parent_t * parent;
  static std::vector<vertex_t *> vertex;
  static std::vector<bound_t *> boundary;
  static std::vector<child_t *> child;
 public:
  HGeometry();
  virtual ~HGeometry() {}

  bool isRefined() const { return false; }
  void refine() {}
};



template <int DOW=1>
  class HGeometry<1,DOW> : public HGeometryInfo<1>, public HGeometryBase
  {
  public:
  enum { dim = 1, dow = DOW };
  typedef HGeometry<0,dow> vertex_t;
  typedef HGeometry<dim-1,dow> bound_t;
  typedef HGeometry<dim,dow> this_t;
  typedef this_t child_t;
  typedef this_t parent_t;

  static void (*mid_point)(const Point<DOW>&, 
                           const Point<DOW>&,
                           bmark_t,
                           Point<DOW>&);
  public:
  int index;
  std::vector<vertex_t *> vertex;
  std::vector<bound_t *> boundary;
  parent_t * parent;
  std::vector<HGeometry<1,DOW> *> child;
  bmark_t bmark;
  public:
  HGeometry();
  virtual ~HGeometry() {}
  public:
  bool isRefined() const;
  bool isIncludePoint(const Point<DOW>&) const;
  void refine();
  void checkIntegrity() const;

  friend std::ostream& operator<< <>(std::ostream&, const HGeometry<1,DOW>&);
  };

template <int DOW=2>
  class HGeometry<2,DOW> : public HGeometryInfo<2>, public HGeometryBase
  {
  public:
  enum { dim = 2, dow = DOW };
  typedef HGeometry<0,dow> vertex_t;
  typedef HGeometry<dim-1,dow> bound_t;
  typedef HGeometry<dim,dow> this_t;
  typedef this_t child_t;
  typedef this_t parent_t;

  int index;
  std::vector<vertex_t *> vertex;
  std::vector<bound_t *> boundary;
  parent_t * parent;
  std::vector<child_t *> child;
  bmark_t bmark;
  public:
  HGeometry();
  virtual ~HGeometry() {}
  public:
  bool isRefined() const;
  bool isIncludePoint(const Point<DOW>&) const;
  void refine();
  void checkIntegrity() const;

  /**
   * ?????????????????????????????????????????????????????????DOW=2??????????????????
   */
  static double 
  triangle_area(const Point<DOW>& v0,
                const Point<DOW>& v1,
                const Point<DOW>& v2) {
    return 0.5*((v1[0] - v0[0])*(v2[1] - v0[1]) - 
                (v1[1] - v0[1])*(v2[0] - v0[0]));
  }

  friend std::ostream& operator<< <>(std::ostream&, const HGeometry<2,DOW>&);
  };

template <int DOW=3>
  class HGeometry<3,DOW> : public HGeometryInfo<3>, public HGeometryBase
  {
  public:
  enum { dim = 3, dow = DOW };
  typedef HGeometry<0,dow> vertex_t;
  typedef HGeometry<dim-1,dow> bound_t;
  typedef HGeometry<dim,dow> this_t;
  typedef this_t child_t;
  typedef this_t parent_t;

  static const int	REFINE_MODEL_03 = 0;
  static const int	REFINE_MODEL_14 = 1;
  static const int	REFINE_MODEL_25 = 2;
  int			refine_model;
  public:
  int index;
  std::vector<vertex_t *> vertex;
  std::vector<bound_t *> boundary;
  parent_t * parent;
  std::vector<child_t *> child;
  bmark_t bmark;
  public:
  HGeometry();
  virtual ~HGeometry() {}
  public:
  bool isRefined() const;
  bool isIncludePoint(const Point<DOW>&) const;
  void refine();
  void checkIntegrity() const;

  /**
   * ?????????????????????????????????????????????????????????DOW=3??????????????????
   */
  static double
  tetrahedron_volume(const Point<DOW>& v0,
                     const Point<DOW>& v1,
                     const Point<DOW>& v2,
                     const Point<DOW>& v3) {
    return ((v1[0] - v0[0])*(v2[1] - v0[1])*(v3[2] - v0[2]) +
            (v1[1] - v0[1])*(v2[2] - v0[2])*(v3[0] - v0[0]) +
            (v1[2] - v0[2])*(v2[0] - v0[0])*(v3[1] - v0[1]) -
            (v1[0] - v0[0])*(v2[2] - v0[2])*(v3[1] - v0[1]) -
            (v1[1] - v0[1])*(v2[0] - v0[0])*(v3[2] - v0[2]) -
            (v1[2] - v0[2])*(v2[1] - v0[1])*(v3[0] - v0[0]))/6.;
  }

  friend std::ostream& operator<< <>(std::ostream&, const HGeometry<3,DOW>&);
  };


/**
 * Hierarchy geometry tree. This is the class to manage all those macro elements, as
 * the roots of all those hierarchy geometries.
 */
template <int DIM, int DOW=DIM>
  class HGeometryTree
  {
  public:
  enum { dim = DIM, dow = DOW };

  private:
  typedef HGeometry<DIM,DOW> entry_t;
  public:
  typedef std::list<entry_t *> container_t;
  private:
  container_t root_element;

  bool _is_locked; /**
                    * ????????????????????????????????????????????????????????????????????????
                    * ????????????????????????????????????????????????????????????????????????
                    * ??????????????????????????????????????????????????????????????????+??????
                    * ??????????????????????????????????????????????????????????????????
                    */

  public:
  typedef _Deref_iterator<typename container_t::iterator, entry_t> RootIterator;
  typedef _Deref_iterator<typename container_t::const_iterator, const entry_t> ConstRootIterator;

  typedef HTools Tools;
  public:
  HGeometryTree() : _is_locked(false) {};
  virtual ~HGeometryTree() {clear();};

  protected:
  bool lock() {
    if (_is_locked) return false;
    else {
      _is_locked = true;
      return true;
    }
  }
  void unlock() {
    _is_locked = false;
  }

  public:
  container_t& rootElement() { return root_element; }
  const container_t& rootElement() const { return root_element; }

  unsigned int n_rootElement() const {return root_element.size();}

  RootIterator beginRootElement() {return RootIterator(root_element.begin());}
  RootIterator endRootElement() {return RootIterator(root_element.end());}

  ConstRootIterator beginRootElement() const {return ConstRootIterator(root_element.begin());}
  ConstRootIterator endRootElement() const {return ConstRootIterator(root_element.end());}

  void clear(); // this method need very complex implementation, left for future
  void checkIntegrity();

  bool is_locked() const { return _is_locked; }
  bool& is_locked() { return _is_locked; }
	
  /**
   * This is the routine used to read in mesh data generated by the software "easymesh".
   * For 2 dimensional case only.
   */
  void readEasyMesh(const std::string&);
	
  /**
   * This is the routine used to read in the mesh data in the internal data format.
   */
  void readMesh(const std::string&);

  friend class IrregularMesh<DIM, DOW>;
  };



/**
 * Hierarcy element. Hierarchy element is the basic component of the irregular
 * mesh. The hierarchy tree to construct the irregular mesh is different from the
 * hierarchy geometry tree. It have only the tree of the elements, but no tree
 * of those lower dimensional geometries. In fact, the irregular mesh is a subtree
 * of the hierarchy element geometry tree.
 */
template <int DIM, int DOW=DIM>
  class HElement : public HGeometryInfo<DIM>, public HGeometryBase {
 public:
 enum { dim = DIM, dow = DOW };
 typedef HGeometry<dim,dow> h_element_t;
 typedef HElement<dim,dow> element_t;
 typedef element_t parent_t;
 typedef element_t child_t;

 public:
 typedef int				ElementType;
 static const ElementType		NOT_ACTIVE = -1;
	
 // for 2 dimensional case
 static const ElementType		TRIANGLE = 0;
 static const ElementType		QUADRILATERAL = 1;
	
 // for 3 dimensional case
 static const ElementType		TETRAHEDRON = 0;
 static const ElementType		TWIN_TETRAHEDRON = 1;
 static const ElementType		FOUR_TETRAHEDRON = 2;
 public:
 int					index;
 double					indicator;
 int					value; // default: -1
 h_element_t *				h_element; // default: NULL
 element_t *				parent; // default: NULL
 std::vector<element_t *>		child; // default: NULL
 public:
 HElement();
 HElement(const element_t&);
 virtual ~HElement();
 public:
 element_t& operator=(const element_t&);
 /* bool is_dummy() const { return h_element->is_dummy(); } */
 bool isRefined() const;
 bool isIncludePoint(const Point<DOW>&) const;
 void refine();
 void checkIntegrity() const;

 friend std::ostream& operator<< <>(std::ostream&, const HElement<DIM, DOW>&);
};



/**
 * IrregularMesh is a image of a complete subtree of the hierarchy element
 * geometry tree. It's related with the hierarchy tree and a regular mesh
 * which is generated from itself to used by finite element space. It mainly
 * provides many iterator to used by other routines to access all those
 * elements, or certain selective elements in the mesh. It provides the most
 * important operation to make the irregular mesh a semi-regular mesh --
 * semiregularize, and the operation to construct the related regular
 * mesh -- regularize. The mesh adaptation are mainly depended on those
 * operations.
 */
template <int DIM, int DOW=DIM>
  class IrregularMesh
  {
  public:
  enum { dim = DIM, dow = DOW };
  typedef RegularMesh<DIM,DOW> mesh_t;
  typedef HGeometryTree<DIM,DOW> tree_t;
  typedef IrregularMesh<DIM,DOW> ir_mesh_t;

  protected:
  typedef HGeometry<DIM,DOW> h_element_t;
  typedef HElement<DIM,DOW> element_t;
  typedef std::list<element_t *> container_t;
  typedef HTools Tools;

  private:
  tree_t * geometry_tree; // default: NULL
  container_t root_element; // default: NULL
  mesh_t * regular_mesh; // default: NULL

  public:
  typedef _Deref_iterator<typename container_t::iterator, element_t> RootIterator;
  typedef _Deref_iterator<typename container_t::const_iterator, const element_t> ConstRootIterator;

  typedef RootFirstElementIterator<DIM, DOW> RootFirstIterator;
  typedef ActiveElementIterator<DIM, DOW> ActiveIterator;

  public:
  IrregularMesh();
  explicit IrregularMesh(tree_t&);
  IrregularMesh(const ir_mesh_t&);
  virtual ~IrregularMesh();

  public:
  void clear();
  ir_mesh_t& operator=(const ir_mesh_t&);

  public:
  RootIterator beginRootElement() {return root_element.begin();}
  RootIterator endRootElement() {return root_element.end();}

  ConstRootIterator beginRootElement() const {return root_element.begin();};
  ConstRootIterator endRootElement() const {return root_element.end();};

  RootFirstIterator beginRootFirstElement();
  RootFirstIterator endRootFirstElement();

  ActiveIterator beginActiveElement();
  ActiveIterator endActiveElement();

  public:
  /** 
   * ???????????????????????????????????????????????????????????????????????????????????? is_bare
   * ????????????????????? is_bare ??? true?????????????????????????????????????????????
   * htree ??????????????????????????????????????????????????????????????????????????????????????????
   * ???????????????????????????????????????????????????????????????????????????????????????????????????
   * ???????????????????????????????????????????????????????????????????????????
   * 
   * @param htree ???????????????
   * @param is_bare ???????????????????????????????????????????????????????????????????????????
   *
   */
  void reinit(tree_t& htree, bool is_bare = false);
  tree_t& geometryTree() const {return *geometry_tree;};
  container_t& rootElement() {return root_element;};
  mesh_t& regularMesh() {return *regular_mesh;};
  const mesh_t& regularMesh() const {return *regular_mesh;};

  virtual void semiregularize();
  void regularize(bool renumerate = true);
  void globalRefine(unsigned int i = 1);
  void randomRefine(double percent = 50.0);
  void writeFormatted(const std::string&);

  void copyTree(const ir_mesh_t&);
  void copyNonnegtiveSubtree(const ir_mesh_t&);

  void copyTree(const element_t *, element_t *);
  void copyNonnegtiveSubtree(const element_t *, element_t *);

  void deleteTree(element_t *);

  friend std::ostream& operator<< <>(std::ostream&, IrregularMesh<DIM, DOW>&);
  friend class IrregularMeshPair<DIM, DOW>;

  protected:
  void checkIntegrity();
  void setGeometryTree(tree_t *);

  void semiregularizeHelper(bool&, int&);
  void semiregularizeHelper(bool&, element_t&, int&);

  void prepareSemiregularize();
  void prepareSemiregularizeHelper(h_element_t *);

  void renumerateElement();

  void refineElement(element_t& h_ele);

  public:
  friend class RegularMesh<DIM, DOW>;
  };



/**
 * This is a derativate class of the \p{class Mesh} in the finite element
 * library. It's used to build the relationship of a mesh used by the finite
 * element mesh and the depended semi-regular mesh.
 */
template <int DIM, int DOW=DIM>
  class RegularMesh : public Mesh<DIM,DOW>
  {
  public:
  enum { dim = DIM, dow = DOW };
  typedef IrregularMesh<DIM,DOW> ir_mesh_t;
  typedef RegularMesh<DIM,DOW> mesh_t;

  private:
  ir_mesh_t * irregular_mesh;

#ifdef __SERIALIZATION__
  std::vector<std::vector<HGeometryBase*> > h_geometry_ptr; /// ????????????HGeometry??????????????????
#endif

  private: /// ????????????????????????????????????????????????????????????????????????????????????
  RegularMesh() {}
  RegularMesh(const mesh_t& m) {}
  mesh_t& operator=(const mesh_t& m) {}
  explicit RegularMesh(ir_mesh_t * m) : irregular_mesh(m) {};
  public:
  ir_mesh_t& irregularMesh() const {return *irregular_mesh;};
  void renumerateElement() {irregular_mesh->renumerateElement();};

#ifdef __SERIALIZATION__
  std::vector<std::vector<HGeometryBase*> >& h_geometry() { return h_geometry_ptr;}
  const std::vector<std::vector<HGeometryBase*> >& h_geometry() const { return h_geometry_ptr;}
  HGeometryBase * h_geometry(int dim, u_int idx) const { return h_geometry_ptr[dim][idx]; }

  /**
   * ?????? GDIM ????????? idx ??????????????? HGeometry ????????????
   */
  template <int GDIM> HGeometry<GDIM,DOW> *
  h_geometry(u_int idx) const { 
    return (HGeometry<GDIM,DOW> *)h_geometry_ptr[GDIM][idx]; 
  }

  /** 
   * ?????? dim ????????? idx ?????????????????????????????????
   * 
   * @param dim ??????
   * @param idx ???????????????
   * @param pid ????????? ID
   * 
   * @return ???????????????????????????
   */
  template <class T, int GDIM> T * 
  new_property(u_int idx, const property_id_t<T>& pid) const {
    return this->template h_geometry<GDIM>(idx)->new_property(pid);
  }
  /** 
   * ?????? dim ????????? idx ???????????????????????????
   * 
   * @param dim ??????
   * @param idx ???????????????
   * @param pid ????????? ID
   * 
   * @return ???????????????
   */
  template <class T, int GDIM> T * 
  get_property(int idx, const property_id_t<T>& pid) const {
    return this->template h_geometry<GDIM>(idx)->get_property(pid);
  }
  template <class T, int GDIM> void
  free_property(int idx, const property_id_t<T>& pid) const {
    this->template h_geometry<GDIM>(idx)->free_property(pid);
  }

  /**
   * ?????? dim ??????????????? geo ?????????????????????
   */
  template <class T, int GDIM> T * 
  new_property(const GeometryBM& geo, const property_id_t<T>& pid) const {
    return this->template h_geometry<GDIM>(geo.index())->new_property(pid);
  }
  /**
   * ?????? dim ??????????????? geo ???????????????
   */
  template <class T, int GDIM> T * 
  get_property(const GeometryBM& geo, const property_id_t<T>& pid) const {
    return this->template h_geometry<GDIM>(geo.index())->get_property(pid);
  }
  template <class T, int GDIM> void
  free_property(const GeometryBM& geo, const property_id_t<T>& pid) const {
    this->template h_geometry<GDIM>(geo.index())->free_property(pid);
  }

  /**
   * ?????? dim ??????????????? geo ?????????????????????
   */
  template <class T> T *
  new_property(int dim, 
               const GeometryBM& geo, 
               const property_id_t<T>& pid) const {
    switch(dim) {
    case 0: return this->template new_property<T,0>(geo, pid);
    case 1: return this->template new_property<T,1>(geo, pid);
    case 2: return this->template new_property<T,2>(geo, pid);
    case 3: return this->template new_property<T,3>(geo, pid);
    }
  }
  /** 
   * ?????? dim ????????? idx ???????????????????????????
   * 
   * @param dim ??????
   * @param idx ???????????????
   * @param pid ????????? ID
   * 
   * @return ???????????????
   */
  template <class T> T *
  get_property(int dim, 
               int idx, 
               const property_id_t<T>& pid) const {
    switch(dim) {
    case 0: return this->template get_property<T,0>(idx, pid);
    case 1: return this->template get_property<T,1>(idx, pid);
    case 2: return this->template get_property<T,2>(idx, pid);
    case 3: return this->template get_property<T,3>(idx, pid);
    }
  }
  template <class T> void
  free_property(int dim, 
                         int idx, 
                         const property_id_t<T>& pid) const {
    switch(dim) {
    case 0: this->template free_property<T,0>(idx, pid); break;
    case 1: this->template free_property<T,1>(idx, pid); break;
    case 2: this->template free_property<T,2>(idx, pid); break;
    case 3: this->template free_property<T,3>(idx, pid); break;
    }
  }

  /** 
   * ?????? dim ????????? idx ?????????????????????????????????
   * 
   * @param dim ??????
   * @param idx ???????????????
   * @param pid ????????? ID
   * 
   * @return ???????????????????????????
   */
  template <class T> T *
  new_property(int dim, 
               int idx, 
               const property_id_t<T>& pid) const {
    switch(dim) {
    case 0: return this->template new_property<T,0>(idx, pid);
    case 1: return this->template new_property<T,1>(idx, pid);
    case 2: return this->template new_property<T,2>(idx, pid);
    case 3: return this->template new_property<T,3>(idx, pid);
    }
  }
  /**
   * ?????? dim ??????????????? geo ?????????????????????
   */
  template <class T> T *
  get_property(int dim, 
               const GeometryBM& geo, 
               const property_id_t<T>& pid) const {
    switch(dim) {
    case 0: return this->template get_property<T,0>(geo, pid);
    case 1: return this->template get_property<T,1>(geo, pid);
    case 2: return this->template get_property<T,2>(geo, pid);
    case 3: return this->template get_property<T,3>(geo, pid);
    }
  }
  template <class T> void
  free_property(int dim, 
                const GeometryBM& geo, 
                const property_id_t<T>& pid) const {
    switch(dim) {
    case 0: this->template free_property<T,0>(geo, pid); break;
    case 1: this->template free_property<T,1>(geo, pid); break;
    case 2: this->template free_property<T,2>(geo, pid); break;
    case 3: this->template free_property<T,3>(geo, pid); break;
    }
  }

#endif // __SERIALIZATION__

  template <class LABEL>
  void refineLabelled(LABEL& flag) {
    typename ir_mesh_t::ActiveIterator
    the_ele = irregular_mesh->beginActiveElement(),
    end_ele = irregular_mesh->endActiveElement();
    for (;the_ele != end_ele;) {
      typename ir_mesh_t::ActiveIterator it = the_ele;
      ++ the_ele;
      if (flag[it->index]) {
        irregular_mesh->refineElement(*it);
      }
    }
  }
  template <class LABEL>
  void coarsenLabelled(LABEL& flag) {
    typedef HElement<DIM,DOW> h_element_t;
    std::list<h_element_t *> coarsen_list;
    property_id_t<int> pid;
    new_property_id(pid);
    typename ir_mesh_t::ActiveIterator
    the_ele = irregular_mesh->beginActiveElement(),
    end_ele = irregular_mesh->endActiveElement();
    for (;the_ele != end_ele;++ the_ele) {
      if (flag[the_ele->index]) {
        HElement<DIM,DOW> * p_ele = the_ele->parent;
        if (p_ele != NULL) {
          int * p_prp = p_ele->get_property(pid);
          if (p_prp == NULL) {
            p_prp = p_ele->new_property(pid);
            *p_prp = 1;
          } else {
            *p_prp += 1;
          }
          if (*p_prp == p_ele->n_child) {
            coarsen_list.push_back(p_ele);
            p_ele->free_property(pid);
          }
        }
      }
    }
    free_property_id(pid);

    typename std::list<h_element_t *>::iterator
    the_ele_ptr = coarsen_list.begin(),
    end_ele_ptr = coarsen_list.end();
    for (;the_ele_ptr != end_ele_ptr;++ the_ele_ptr) {
      (*the_ele_ptr)->value = 0;
    }
  } 

  void renumerateElementHSFC(void (*f)(const double*, double*)=NULL);
  /**
   * Write the mesh data into \p{easymesh} format data files. For 2-d only.
   */
  virtual void writeEasyMesh(const std::string&) const;
  /**
   * Write the mesh data into \p{Techplot} Finite-Element data format.
   */
  virtual void writeTecplotData(const std::string&) const;
  /**
   * Write the mesh data into \p{IBM OpenDX} native data format.
   */
  virtual void writeOpenDXData(const std::string&) const;
  /**
   * ????????? SimplestMesh ??????????????????????????????????????????????????????????????????
   * ????????????????????????????????????????????????????????????????????????????????????????????????
   * ????????? SimplestSimplexMesh ?????????????????????????????????????????????????????????
   * ??????????????????????????????????????????????????????????????????
   */
  virtual void writeSimplestSimplexMesh(const std::string&) const;
  /**
   * ?????????????????? mesh ????????????????????????????????????????????????
   */
  virtual void writeSimplexMesh(const std::string&) const;
  public:
  friend class IrregularMesh<DIM,DOW>;
  };

/**
 * This is the base class for some other iterators to access those elements in the irregular
 * mesh. The elements of the irregular mesh are divided into two class: active and nonactive.
 * The active elements are those leaf nodes of the irregular mesh while the nonactive
 * are not leaves. Because the integration on the domain is an opearation on all those
 * active element, a iterator provided for such routines.
 */
template <int DIM, int DOW=DIM>
  class ElementIterator
  {
  public:
  enum { n_child = HGeometry<DIM,DOW>::n_child };

  typedef HElement<DIM,DOW> value_t;
  typedef value_t * pointer_t;
  typedef value_t& reference_t;

  typedef typename std::list<pointer_t>::iterator root_t;
  typedef ElementIterator<DIM,DOW> this_t;
  typedef IrregularMesh<DIM, DOW> ir_mesh_t;

  protected:
  ir_mesh_t * mesh;
  root_t root_element;
  pointer_t element;

  public:
  ElementIterator();
  ElementIterator(ir_mesh_t *, root_t&, pointer_t);
  ElementIterator(const this_t&);
  virtual ~ElementIterator();

  public:
  this_t& operator=(const this_t&);

  const reference_t operator*() const {return *element;};
  reference_t operator*() {return *element;};

  operator const pointer_t() const {return element;};
  operator pointer_t() {return element;};

  const pointer_t operator->() const {return element;};
  pointer_t operator->() {return element;};

  virtual this_t& operator++() = 0;

  friend bool operator== <>(const this_t&, this_t&);
  friend bool operator!= <>(const this_t&, this_t&);

  public:
  friend class IrregularMesh<DIM, DOW>;
  friend class HElement<DIM, DOW>;
  friend class ActiveElementPairIterator<DIM, DOW>;
  };



/**
 * Iterator to all those elements in the hierarchy tree. This iterator access the parent
 * node before its children.
 */
template <int DIM, int DOW=DIM>
  class RootFirstElementIterator : public ElementIterator<DIM, DOW>
  {
  public:
  enum { n_child = HGeometry<DIM,DOW>::n_child };

  typedef ElementIterator<DIM,DOW> base_t;
  typedef typename base_t::root_t root_t;
  typedef RootFirstElementIterator<DIM,DOW> this_t;

  using base_t::mesh;
  using base_t::root_element;
  using base_t::element;

  public:
  RootFirstElementIterator() {};
  RootFirstElementIterator(IrregularMesh<DIM, DOW> * m, 
                           root_t& i, 
                           HElement<DIM, DOW> * e) :
  base_t::ElementIterator(m, i, e) {};
  public:
  virtual this_t& operator++();
  };



/**
 * Iterator to active elements in the hierarchy tree. This is a derivative class of
 * \p{RootFirstElementIterator}, and the \p{operator++} is overrided.
 */
template <int DIM, int DOW=DIM>
  class ActiveElementIterator : public RootFirstElementIterator<DIM, DOW>
  {
  public:
  enum { n_child = HGeometry<DIM,DOW>::n_child };

  typedef RootFirstElementIterator<DIM,DOW> base_t;
  typedef typename base_t::root_t root_t;
  typedef ActiveElementIterator<DIM,DOW> this_t;
  public:
  ActiveElementIterator() {};
  ActiveElementIterator(IrregularMesh<DIM, DOW> * m, 
                        root_t& i,
                        HElement<DIM, DOW> * e) : base_t(m, i, e) {};
  ActiveElementIterator(const base_t& it) : base_t(it) {}
  public:
  virtual this_t& operator++();
  };



/**
 * This class is a packing to irregular mesh. The task of this class is to provide
 * an environment for a so-called "element pair iterator" to access all those
 * active element in the union of the two irregular mesh.
 */
template <int DIM, int DOW=DIM>
  class IrregularMeshPair
  {
  public:
  enum { dim = DIM, dow = DOW };

  typedef IrregularMesh<DIM, DOW> ir_mesh_t;
  typedef ActiveElementPairIterator<DIM, DOW> iterator;

  ir_mesh_t *	mesh0;
  ir_mesh_t *	mesh1;
  public:
  IrregularMeshPair(ir_mesh_t&, ir_mesh_t&);
  IrregularMeshPair(ir_mesh_t *, ir_mesh_t *);
  ~IrregularMeshPair();
  public:
  iterator beginActiveElementPair();
  iterator endActiveElementPair();
  };



/**
 * This is the iterator to get those active element pair. This class is the
 * iterator used for those opeartions on the whole domain but two meshes related, such as
 * the $L^2$ inner product of two finite element functions on the two different meshes.
 * It's the pack of two different iterators on the two different meshes and its one variable
 * shows the relationship of the two active element in the two different meshes.
 */
template <int DIM, int DOW=DIM>
  class ActiveElementPairIterator
  {
  public:
  typedef IrregularMeshPair<DIM, DOW> ir_mesh_pair_t;

  public:
  typedef int State;
  static const State GREAT_THAN			= -1;	// 0 is the ancestor of 1
  static const State EQUAL			= 0;	// equal
  static const State LESS_THAN			= 1;	// 0 is a descendant of 1

  private:
  typedef RootFirstElementIterator<DIM,DOW> iterator;
  typedef ActiveElementPairIterator<DIM,DOW> this_t;

  public:
  typedef HElement<DIM,DOW> value_t;
  typedef value_t& reference_t;
  typedef value_t * pointer_t;

  ir_mesh_pair_t * mesh_pair;
  State						st;
  iterator		iterator0;
  iterator		iterator1;

  public:
  ActiveElementPairIterator() : mesh_pair(NULL) {};
  ActiveElementPairIterator(ir_mesh_pair_t * mp,
                            State s,
                            const iterator& it0,
                            const iterator& it1) :
  mesh_pair(mp), st(s), iterator0(it0), iterator1(it1) {};
  ActiveElementPairIterator(const this_t&);
  ~ActiveElementPairIterator() {};

  public:
  const reference_t operator()(u_int i) const {
    if (i == 0) return *iterator0;
    else if (i == 1) return *iterator1;
    else Assert (false, ExcInternalError()); // something must be wrong
  }
  reference_t operator()(u_int i) {
    if (i == 0) return *iterator0;
    else if (i == 1) return *iterator1;
    else {
      Assert (false, ExcInternalError()); // something must be wrong
      return *((HElement<DIM, DOW> *)NULL);
    }
  };

  const State& state() const {return st;};

  this_t& operator=(const this_t&);
  this_t& operator++();

  friend bool operator== <>(const this_t&, this_t&);
  friend bool operator!= <>(const this_t&, this_t&);
  public:
  friend class IrregularMeshPair<DIM, DOW>;
  };



/**
 * Indicator of the mesh adaptor, related with a mesh. It's in fact a \p{double} array
 * corresponding to every element of mesh, to indicate if the element should be
 * refined or coarsed. The package will check if the indicator is related with certain
 * mesh before applying it to adapt the mesh.
 */
template <int DIM, int DOW=DIM>
  class Indicator : public std::vector<double>
  {
  public:
  enum { dim = DIM, dow = DOW };
  typedef RegularMesh<DIM, DOW> mesh_t;

  public:
  mesh_t *				msh;
  public:
  Indicator() : msh(NULL) {};
  explicit Indicator(mesh_t& m) : msh(&m) {
    resize(msh->n_geometry(DIM));
  };
  ~Indicator() {};
  public:
  const mesh_t& mesh() const {return *msh;}
  void reinit(mesh_t& m, bool is_bare = false) {
    msh = &m;
    if (! is_bare) {
      resize(msh->n_geometry(DIM));
      std::fill(begin(), end(), 0.0);
    }
  }
  };



/**
 * MeshAdaptor is the class used to implement mesh adaptation. It is designed
 * to applied on two mesh - to adapt one(\p{from mesh}) according the indicator
 * and store the result into another mesh(\p{to mesh}). If the \p{from mesh} is
 * the same as the \p{to mesh}, the class then will adapt the mesh itself. The
 * adaptation operation is divided into three steps: (1) prepare the \p{to mesh},
 * in fact copy the \p{from mesh} to the \p{to mesh} if they are different; (2)
 * collect the indicator, it colloect the indicator from the leaf nodes of the
 * mesh to those non-leaf nodes according the \p{convergence order}; (3) adapt
 * the mesh adaptation, by getting a irregular mesh at first and then adopt the
 * semiregularization to the irregular mesh to obtain a semiregular mesh. This
 * class also provide a facility to global refine a mesh.
 */
template <int DIM, int DOW=DIM>
  class MeshAdaptor
  {
  public:
  enum { dim = DIM, dow = DOW };
  typedef IrregularMesh<DIM,DOW> ir_mesh_t;
  typedef Indicator<DIM,DOW> indicator_t;

  private:
  ir_mesh_t *			from_mesh;
  ir_mesh_t *			to_mesh;
  indicator_t *			ind;
  double			tol;
  double			convergence_order;
  int				refine_step;
  double			refine_threshold;
  double			coarse_threshold;

  bool				_is_refine_only;

  public:
  MeshAdaptor();
  explicit MeshAdaptor(ir_mesh_t& f);
  MeshAdaptor(ir_mesh_t& f, ir_mesh_t& t);
  ~MeshAdaptor();

  public:
  void reinit(ir_mesh_t& f) {from_mesh = &f; to_mesh = &f;};
  void reinit(ir_mesh_t& f, ir_mesh_t& t)  {from_mesh = &f; to_mesh = &t;};
  const ir_mesh_t& fromMesh() const {return *from_mesh;};
  void setFromMesh(ir_mesh_t& f) {from_mesh = &f;};
  const ir_mesh_t& toMesh() const {return *to_mesh;};
  void  setToMesh(ir_mesh_t& t) {to_mesh = &t;};
  const indicator_t& indicator() const {return *ind;};
  const double& indicator(const int& i) const {return (*ind)[i];};
  double& indicator(const int& i) {return (*ind)[i];};
  void setIndicator(indicator_t& i) {
    ind = &i;
    Assert (&(ind->mesh()) == &(from_mesh->regularMesh()), ExcInternalError());
  };
  const double& tolerence() const {return tol;};
  double& tolerence() {return tol;};
  const double& convergenceOrder() const {return convergence_order;};
  double& convergenceOrder() {return convergence_order;};
  const int& refineStep() const {return refine_step;};
  int& refineStep() {return refine_step;};
  const double& refineThreshold() const {return refine_threshold;};
  double& refineThreshold() {return refine_threshold;};
  const double& coarseThreshold() const {return coarse_threshold;};
  double& coarseThreshold() {return coarse_threshold;};

  bool is_refine_only() const { return _is_refine_only; }
  bool& is_refine_only() { return _is_refine_only; }

  bool is_indicator_underflow(double ind) const {
    return (ind < coarse_threshold*tolerence());
  }
  bool is_indicator_overflow(double ind) const {
    double convergence_coefficient = pow(2.0, DIM + convergenceOrder());
    return (ind > refine_threshold*convergence_coefficient*tolerence());
  }
  bool is_indicator_overflow(double ind, double convergence_coefficient) const {
    return (ind > refine_threshold*convergence_coefficient*tolerence());
  }

  public:
  void globalRefine(unsigned int i = 1);
  void randomRefine(double percent = 50.0);
  void adapt();
  private:
  void collectIndicator(HElement<DIM,DOW>&, double);
  void collectIndicator();
  void prepareToMesh();
  void implementAdaption();
  void adaptElement(HElement<DIM,DOW>&, double, int);
  };

AFEPACK_CLOSE_NAMESPACE

#endif // _HGeometry_h_

/**
 * end of file
 * 
 */

