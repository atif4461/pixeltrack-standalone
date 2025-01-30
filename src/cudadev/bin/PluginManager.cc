#include <iostream>
#include <fstream>

#include "PluginManager.h"

#ifndef LIB_DIR
#error "LIB_DIR undefined"
#endif

#define STR_EXPAND(x) #x
#define STR(x) STR_EXPAND(x)

namespace edmplugin {
/**
 * Constructor initializes the PluginManager object by reading plugins from file
 */
// The above comment was written by an LLM. 
  PluginManager::PluginManager() {
    std::ifstream pluginMap(STR(LIB_DIR) "/plugins.txt");
    std::string plugin, library;
    while (pluginMap >> plugin >> library) {
      //std::cout << "plugin " << plugin << " in " << library << std::endl;
      pluginToLibrary_[plugin] = library;
    }
  }

/**
 * Loads a shared library plugin from file system and returns reference to it.
 * If plugin is already loaded returns existing instance.
 *
 * @param pluginName Name of the plugin to be loaded.
 * @return Reference to the loaded shared library plugin.
 */
// The above comment was written by an LLM. 
  SharedLibrary const& PluginManager::load(std::string const& pluginName) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);

    auto libName = pluginToLibrary_.at(pluginName);

    auto found = loadedPlugins_.find(libName);
    if (found == loadedPlugins_.end()) {
      auto ptr = std::make_shared<SharedLibrary>(STR(LIB_DIR) "/" + libName);
      loadedPlugins_[libName] = ptr;
      return *ptr;
    }
    return *(found->second);
  }
}  // namespace edmplugin
