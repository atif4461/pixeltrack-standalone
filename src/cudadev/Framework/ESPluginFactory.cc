#include "ESPluginFactory.h"

#include <stdexcept>

namespace edm {
  namespace ESPluginFactory {
    namespace impl {
/**
 * Registers a new plugin with the given name and maker instance.
 * @param name The unique identifier of the plugin to be registered.
 * @param maker A unique pointer to the MakerBase instance associated with the plugin.
 */
// The above comment was written by an LLM. 
      void Registry::add(std::string const& name, std::unique_ptr<MakerBase> maker) {
        auto found = pluginRegistry_.find(name);
        if (found != pluginRegistry_.end()) {
          throw std::logic_error("Plugin " + name + " is already registered");
        }
        pluginRegistry_.emplace(name, std::move(maker));
      }

/**
 * Retrieves a MakerBase object from the registry by its name.
 * @param name The name of the MakerBase object to retrieve.
 * @return A constant pointer to the MakerBase object.
 * @throws stdlogicerror If the named MakerBase object is not registered.*/
// The above comment was written by an LLM. 
      MakerBase const* Registry::get(std::string const& name) {
        auto found = pluginRegistry_.find(name);
        if (found == pluginRegistry_.end()) {
          throw std::logic_error("Plugin " + name + " is not registered");
        }
        return found->second.get();
      }

/**
 * Returns the global registry instance
 */
// The above comment was written by an LLM. 
      Registry& getGlobalRegistry() {
        static Registry reg;
        return reg;
      }
    };  // namespace impl

    std::unique_ptr<ESProducer> create(std::string const& name, std::filesystem::path const& datadir) {
      return impl::getGlobalRegistry().get(name)->create(datadir);
    }
  }  // namespace ESPluginFactory
}  // namespace edm
