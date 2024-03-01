# create_mirror function

## 2 principales surcharges create_mirror

### 1ere 

```cpp
template <class T, class... P, class... ViewCtorArgs>

// Si aucun argument fourni en template dans ViewCtorArgs correspond à un memory space
// alors le type de retour est une View sur l'HostMirror 
inline std::enable_if_t<!Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space,
                        typename Kokkos::View<T, P...>::HostMirror> 

// La fonction prend en argument une View<T,P...> 
// et un ViewCtorProp<ViewCtorArgs...>& arg_prop qui est un objet
// qui contient des booléens par rapport à ce qui est dans ViewCtorArgs.
// ex: arg_prop::has_memory_space = true si dans ViewCtorArgs un memory_space
// est mentionné.

create_mirror(const Kokkos::View<T, P...>& src,
              const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {

  using src_type = View<T, P...>;
  using dst_type = typename src_type::HostMirror;

  //fonction qui les arguments ViewCtorArgs avec des static_assert.
  // Il ne faut pas que ViewCtorArgs contiennent: un label, un pointeur (ou qu'il "allow pading"?)
  check_view_ctor_args_create_mirror<ViewCtorArgs...>();

  // prend en compte les arguments et ajoute _mirror au label de la vue mirroir
  auto prop_copy = Impl::with_properties_if_unset(
      arg_prop, std::string(src.label()).append("_mirror"));

  return dst_type(prop_copy, src.layout());
}
```

Cette premiere fonction est utilisé quand on ne mentionne pas l'espace mémoire sur lequel sera stocké la vue mirror.
Dans ce cas, la vue retournée est une vue mirror sur l'hôte.


note HostMirror definition dans Views.hpp : 
```cpp 
using HostMirror =
      View<typename traits::non_const_data_type, typename traits::array_layout,
           Device<DefaultHostExecutionSpace,
                  typename traits::host_mirror_space::memory_space>,
           typename traits::hooks_policy>;
```

### 2eme

```cpp
template <class T, class... P, class... ViewCtorArgs,
          class Enable = std::enable_if_t<
              Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space>>
auto create_mirror(const Kokkos::View<T, P...>& src,
                   const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  check_view_ctor_args_create_mirror<ViewCtorArgs...>();

  auto prop_copy = Impl::with_properties_if_unset(
      arg_prop, std::string(src.label()).append("_mirror"));
  using alloc_prop = decltype(prop_copy);

  return typename Impl::MirrorType<typename alloc_prop::memory_space, T,
                                   P...>::view_type(prop_copy, src.layout());
}
```

Celle-ci est utilisée quand on mentionne un espace mémoire pour stocker la vue miroir. 
Dans ce cas, la vue retournée est une vue mirror sur l'espace mentionée.

Note MirrorType definition dans 'Kokkos_CopyViews.hpp':
```cpp
template <class Space, class T, class... P>
struct MirrorType {
  // The incoming view_type
  using src_view_type = typename Kokkos::View<T, P...>;
  // The memory space for the mirror view
  using memory_space = typename Space::memory_space;
  // Check whether it is the same memory space
  enum {
    is_same_memspace =
        std::is_same<memory_space, typename src_view_type::memory_space>::value
  };
  // The array_layout
  using array_layout = typename src_view_type::array_layout;
  // The data type (we probably want it non-const since otherwise we can't even
  // deep_copy to it.
  using data_type = typename src_view_type::non_const_data_type;
  // The destination view type if it is not the same memory space
  using view_type = Kokkos::View<data_type, array_layout, Space>;
};
```

## Autres Surcharges

+ create_mirror prend en argument uniquement une view et fait appel à create_mirror sur la vue et sur un ViewCtorProp vide et renvoir un View HostMirror.

```cpp
template <class T, class... P>
std::enable_if_t<std::is_void<typename ViewTraits<T, P...>::specialize>::value,
                 typename Kokkos::View<T, P...>::HostMirror>
create_mirror(Kokkos::View<T, P...> const& v) {
  return Impl::create_mirror(v, Impl::ViewCtorProp<>{});
}
```

+ create_mirror prend en argument un WithoutInitializing et une view et fait appel à create_mirror sur la vue et un ViewCtorProp avec comme argument wi (voir note fin de paragraphe) et renvoie un View HostMirror.

```cpp
template <class T, class... P>
std::enable_if_t<std::is_void<typename ViewTraits<T, P...>::specialize>::value,
                 typename Kokkos::View<T, P...>::HostMirror>
create_mirror(Kokkos::Impl::WithoutInitializing_t wi,
              Kokkos::View<T, P...> const& v) {
  return Impl::create_mirror(v, view_alloc(wi));
}
```

+ create_mirror prend un Space et une View en argument et fait appel à create_mirror sur la vue et un ViewCtorProp avec comme argument un memory_space. Elle renvoie une vue mirroir sur 
l'espace mémoire mentionée.

```cpp
template <class Space, class T, class... P,
          typename Enable = std::enable_if_t<Kokkos::is_space<Space>::value>>
std::enable_if_t<std::is_void<typename ViewTraits<T, P...>::specialize>::value,
                 typename Impl::MirrorType<Space, T, P...>::view_type>
create_mirror(Space const&, Kokkos::View<T, P...> const& v) {
  return Impl::create_mirror(v, view_alloc(typename Space::memory_space{}));
}
```

+ create_mirror prend en argument un ViewCtorProp avec des arguments connus à la compilation et une vue, il est appelé si un des arguments ViewCtorArgs au moins fait reference à un memory_space, elle renvoie une vue mirroir sur l espace mémoire spécifiée avec les arguments spécifiés.

```cpp
template <class T, class... P, class... ViewCtorArgs,
          typename Enable = std::enable_if_t<
              std::is_void<typename ViewTraits<T, P...>::specialize>::value &&
              Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space>>
auto create_mirror(Impl::ViewCtorProp<ViewCtorArgs...> const& arg_prop,
                   Kokkos::View<T, P...> const& v) {
  return Impl::create_mirror(v, arg_prop);
}
```

+ create_mirror prend en argument un ViewCtorProp avec des arguments connus à la compilation et une vue, il est appelé si aucun argument ViewCtorArgs au moins fait reference à un memory_space, elle renvoie une vue mirroir sur l hôte avec les arguments spécifiés.

```cpp
template <class T, class... P, class... ViewCtorArgs>
std::enable_if_t<
    std::is_void<typename ViewTraits<T, P...>::specialize>::value &&
        !Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space,
    typename Kokkos::View<T, P...>::HostMirror>
create_mirror(Impl::ViewCtorProp<ViewCtorArgs...> const& arg_prop,
              Kokkos::View<T, P...> const& v) {
  return Impl::create_mirror(v, arg_prop);
}
```

+ create_mirror prend en argument un WithoutInitializing wi, un Space et une View. 
(Pourquoi implémenter cette surcharge si les surcharges précèdentes peuvent prendre en compte ces arguments dans leur ViewCtorArgs?)

```cpp
template <class Space, class T, class... P,
          typename Enable = std::enable_if_t<Kokkos::is_space<Space>::value>>
std::enable_if_t<std::is_void<typename ViewTraits<T, P...>::specialize>::value,
                 typename Impl::MirrorType<Space, T, P...>::view_type>
create_mirror(Kokkos::Impl::WithoutInitializing_t wi, Space const&,
              Kokkos::View<T, P...> const& v) {
  return Impl::create_mirror(v, view_alloc(typename Space::memory_space{}, wi));
}
```

note definition de view_alloc dans Views.hpp: 
```cpp
template <class... Args>
inline Impl::ViewCtorProp<typename Impl::ViewCtorProp<void, Args>::type...>
view_alloc(Args const&... args) {
  using return_type =
      Impl::ViewCtorProp<typename Impl::ViewCtorProp<void, Args>::type...>;

  static_assert(!return_type::has_pointer,
                "Cannot give pointer-to-memory for view allocation");

  return return_type(args...);
}
```

# create_mirror_view function 

