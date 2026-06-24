NEURON {
  ARTIFICIAL_CELL VecStim
}

ASSIGNED {
  index
  etime (ms)
  space
}

VERBATIM
extern double* vector_vec();
extern int vector_capacity();
extern void* vector_arg();
ENDVERBATIM

INITIAL {
  index = 0
  element()
  if (index > 0) {
    net_send(etime - t, 1)
  }
}

PROCEDURE element() {
VERBATIM
  { void* vv; int i, size; double* px;
    i = (int)index;
    if (i >= 0) {
      vv = *((void**)(&space));
      if (vv) {
        size = vector_capacity(vv);
        px = vector_vec(vv);
        if (i < size) {
          etime = px[i];
          index = i + 1;
        }else{
          index = -1;
        }
      }else{
        index = -1;
      }
    }
  }
ENDVERBATIM
}

NET_RECEIVE (w) {
  if (flag == 0) {
    net_event(t)
    element()
    if (index > 0) {
      net_send(etime - t, 1)
    }
  }
}

PROCEDURE play() {
VERBATIM
  void** vv;
  vv = (void**)(&space);
  *vv = nullptr;
  if (ifarg(1)) {
    *vv = vector_arg(1);
  }
ENDVERBATIM
}