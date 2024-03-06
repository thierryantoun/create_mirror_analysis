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

1. create_mirror prend en argument uniquement une view et fait appel à create_mirror sur la vue et sur un ViewCtorProp vide et renvoir un View HostMirror.

```cpp
template <class T, class... P>
std::enable_if_t<std::is_void<typename ViewTraits<T, P...>::specialize>::value,
                 typename Kokkos::View<T, P...>::HostMirror>
create_mirror(Kokkos::View<T, P...> const& v) {
  return Impl::create_mirror(v, Impl::ViewCtorProp<>{});
}
```

2. create_mirror prend en argument un WithoutInitializing et une view et fait appel à create_mirror sur la vue et un ViewCtorProp avec comme argument wi (voir note fin de paragraphe) et renvoie un View HostMirror.

```cpp
template <class T, class... P>
std::enable_if_t<std::is_void<typename ViewTraits<T, P...>::specialize>::value,
                 typename Kokkos::View<T, P...>::HostMirror>
create_mirror(Kokkos::Impl::WithoutInitializing_t wi,
              Kokkos::View<T, P...> const& v) {
  return Impl::create_mirror(v, view_alloc(wi));
}
```

3. create_mirror prend un Space et une View en argument et fait appel à create_mirror sur la vue et un ViewCtorProp avec comme argument un memory_space. Elle renvoie une vue mirroir sur 
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

4. create_mirror prend en argument un ViewCtorProp avec des arguments connus à la compilation et une vue, il est appelé si un des arguments ViewCtorArgs au moins fait reference à un memory_space, elle renvoie une vue mirroir sur l espace mémoire spécifiée avec les arguments spécifiés.

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

5. create_mirror prend en argument un ViewCtorProp avec des arguments connus à la compilation et une vue, il est appelé si aucun argument ViewCtorArgs au moins fait reference à un memory_space, elle renvoie une vue mirroir sur l hôte avec les arguments spécifiés.

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

6. create_mirror prend en argument un WithoutInitializing wi, un Space et une View. 
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

## 4 Surcharges principales de la fonction 

1. Cette première surcharge renvoie juste une vue qui a les caractéristiques d'une View::HostMirror.

```cpp
template <class T, class... P, class... ViewCtorArgs>

// Si aucun argument de ViewCtorArgs ne fait reference à un memory_space &
// Si le memory_space de la view qui sera prise en argument est la meme que le HostSpace du HostMirrorView
// Et s'ils ont le meme data_type alors on renvoie un Kokkos::View<T, P...>::HostMirror 

inline std::enable_if_t<
    !Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space &&
        (std::is_same<
             typename Kokkos::View<T, P...>::memory_space,
             typename Kokkos::View<T, P...>::HostMirror::memory_space>::value &&
         std::is_same<
             typename Kokkos::View<T, P...>::data_type,
             typename Kokkos::View<T, P...>::HostMirror::data_type>::value),
    typename Kokkos::View<T, P...>::HostMirror>
    
create_mirror_view(const Kokkos::View<T, P...>& src,
                   const Impl::ViewCtorProp<ViewCtorArgs...>&) {
  check_view_ctor_args_create_mirror<ViewCtorArgs...>();
  return src;
}
```

2. Cette surcharge est utilisé quand aucun argument ViewCtorArgs ne fait mention d'un memory space, si le data_type ou le memory_space de la vue en argument diffère de l'HostMirror de cette même vue. Elle retourne une vue miroir de la vue passé en argument sur l'hôte.

```cpp 
template <class T, class... P, class... ViewCtorArgs>

// Si aucun argument de ViewCotArgs ne fait reference à un memory_space 
// Si le memory_space de la View<T, P...> est different du memory_space de l'HostMirror
// Ou si le data_type est différent alors la fonction retournera un View<T,P...>::HostMirror

inline std::enable_if_t<
    !Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space &&
        !(std::is_same<typename Kokkos::View<T, P...>::memory_space,
                       typename Kokkos::View<
                           T, P...>::HostMirror::memory_space>::value &&
          std::is_same<
              typename Kokkos::View<T, P...>::data_type,
              typename Kokkos::View<T, P...>::HostMirror::data_type>::value),
    typename Kokkos::View<T, P...>::HostMirror>
    
create_mirror_view(const Kokkos::View<T, P...>& src,
                   const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
                   
  // Cette surcharge fait appel à la 1ere surcharge de create_mirror qui renvoie une Vue mirroir sur l'hôte
  
  return Kokkos::Impl::create_mirror(src, arg_prop);
} 
```

3. 3eme surcharge qui prend en argument une const view et une reference à un ViewCtorProp. Il renvoie directement la vue en argument si les conditions de compilation sont vérifiées.

```cpp 

// Cette surcharge est appelé si un argument de ViewCtorArgs fait référence à un memory_space. 

template <class T, class... P, class... ViewCtorArgs,
          class = std::enable_if_t<
              Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space>>
              
// Cette surcharge est appelé si la méthode is_same_memspace de MirrorViewType est True (voir note suivante) , dans ce cas, le type retourné est un view_type de la structure MirrorViewType.

std::enable_if_t<Impl::MirrorViewType<
                     typename Impl::ViewCtorProp<ViewCtorArgs...>::memory_space,
                     T, P...>::is_same_memspace,
                 typename Impl::MirrorViewType<
                     typename Impl::ViewCtorProp<ViewCtorArgs...>::memory_space,
                     T, P...>::view_type>
                     
create_mirror_view(const Kokkos::View<T, P...>& src,
                   const Impl::ViewCtorProp<ViewCtorArgs...>&) {
  check_view_ctor_args_create_mirror<ViewCtorArgs...>();
  return src;
}
```

+ Note, structure MirrorViewType dans 'Kokkos_CopyViews.hpp' : 

```cpp 
template <class Space, class T, class... P>
struct MirrorViewType {
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
  using dest_view_type = Kokkos::View<data_type, array_layout, Space>;
  // If it is the same memory_space return the existsing view_type
  // This will also keep the unmanaged trait if necessary
  using view_type =
      std::conditional_t<is_same_memspace, src_view_type, dest_view_type>;
};
```

4. Cette surcharge fait appel à la 2eme surcharge de create_mirror. 
Rappel: Cette surcharge renvoie un ```cpp Impl::MirrorType<typename alloc_prop::memory_space, T, P...>::view_type(prop_copy, src.layout()) ``` alors que cette fonction ci-dessous indique retourner un view_type issu d'un MirrorViewType. Redondance ?

```cpp 
// Memes conditions sfinae que la surcharge precedente.
template <class T, class... P, class... ViewCtorArgs,
          class = std::enable_if_t<
              Impl::ViewCtorProp<ViewCtorArgs...>::has_memory_space>>
              
// Cette fois, si is_same_memspace est faux alors le type retourné est un MirrorViewType::view_type
std::enable_if_t<!Impl::MirrorViewType<
                     typename Impl::ViewCtorProp<ViewCtorArgs...>::memory_space,
                     T, P...>::is_same_memspace,
                 typename Impl::MirrorViewType<
                     typename Impl::ViewCtorProp<ViewCtorArgs...>::memory_space,
                     T, P...>::view_type>
                     
create_mirror_view(const Kokkos::View<T, P...>& src,
                   const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop) {
  return Kokkos::Impl::create_mirror(src, arg_prop);
}
```

## Autres surcharges. 

1. Cette surcharge prend simplement une const View qui a le même memory_space et le même data_type que la View::HostMirror et renvoie cette View qui est de fait une View::HostMirror.

```cpp 
template <class T, class... P>

// Si le memory_space et le data_type entre la View et la View::HostMirror sont les mêmes, le type retourné est un View::HostMirror
std::enable_if_t<
    std::is_same<
        typename Kokkos::View<T, P...>::memory_space,
        typename Kokkos::View<T, P...>::HostMirror::memory_space>::value &&
        std::is_same<
            typename Kokkos::View<T, P...>::data_type,
            typename Kokkos::View<T, P...>::HostMirror::data_type>::value,
    typename Kokkos::View<T, P...>::HostMirror>
create_mirror_view(const Kokkos::View<T, P...>& src) {
  return src;
}
```

2. Cette surcharge prend en argument une const View et fait appel à la surcharge 1. de ##autres-surcharges .

```cpp 
template <class T, class... P>

// Si le View::memory_space est différent que le View::HostMirror::memory_space ou si leur data_type sont différents
// Alors le type de sortie est un View::HostMirror

std::enable_if_t<
    !(std::is_same<
          typename Kokkos::View<T, P...>::memory_space,
          typename Kokkos::View<T, P...>::HostMirror::memory_space>::value &&
      std::is_same<
          typename Kokkos::View<T, P...>::data_type,
          typename Kokkos::View<T, P...>::HostMirror::data_type>::value),
    typename Kokkos::View<T, P...>::HostMirror>
create_mirror_view(const Kokkos::View<T, P...>& src) {
  return Kokkos::create_mirror(src);
}
```
3. Cette surcharge prend en argument un view_alloc sur un WithoutInitializing. Il fait appel à la surcharge 1. ou 2. des principales surcharges de create_mirror_view suivant les paramètres de la View passés en argument. Elle retourne la View::HostMirror avec l'agument WithoutInitializing.

```cpp 
template <class T, class... P>
typename Kokkos::View<T, P...>::HostMirror create_mirror_view(
    Kokkos::Impl::WithoutInitializing_t wi, Kokkos::View<T, P...> const& v) {
  return Impl::create_mirror_view(v, view_alloc(wi));
}
```

4. Cette surcharge prend en argument une View, un paramètre WithoutInitializing et une const reference à un Space. 

```cpp 
// Vérification du template Space.
template <class Space, class T, class... P,
          typename Enable = std::enable_if_t<Kokkos::is_space<Space>::value>>
typename Impl::MirrorViewType<Space, T, P...>::view_type create_mirror_view(
    Kokkos::Impl::WithoutInitializing_t wi, Space const&,
    Kokkos::View<T, P...> const& v) {
  return Impl::create_mirror_view(
      v, view_alloc(typename Space::memory_space{}, wi));
}
```

5. Je ne comprends pas pourquoi on ne garde pas que celle-là ? Si dans ViewCtorArgs peuvent être appelés des WithoutInitializing et des Space, quelle est la plus value des surcharges 3.4. de cette même section par rapport à celle-là ?

```cpp
template <class T, class... P, class... ViewCtorArgs>
auto create_mirror_view(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
                        const Kokkos::View<T, P...>& v) {
  return Impl::create_mirror_view(v, arg_prop);
}
```






























